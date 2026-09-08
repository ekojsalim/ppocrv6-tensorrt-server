//! Cropped upright lines. The native worker is also owned by full-page OCR;
//! its own mutex protects the shared context across both Rust entry points.
use super::*;
use std::collections::BTreeMap;

pub(super) struct LineWorker {
    pub recognizer: Mutex<NativeRecognizer>,
    buckets: Vec<i32>,
    max_batch: i32,
    max_images: usize,
    max_bytes: usize,
    max_pixels: u64,
}

impl LineWorker {
    pub fn new(
        recognizer: NativeRecognizer,
        buckets: &str,
        max_batch: i32,
        max_width: i32,
        max_images: usize,
        max_bytes: usize,
        max_pixels: u64,
    ) -> anyhow::Result<Self> {
        anyhow::ensure!(
            max_images > 0 && max_bytes > 0 && max_pixels > 0,
            "line request limits must be positive"
        );
        let mut buckets = buckets
            .split(',')
            .map(|s| s.trim().parse::<i32>())
            .collect::<Result<Vec<_>, _>>()?;
        anyhow::ensure!(
            buckets.iter().all(|&b| b > 0 && b <= max_width),
            "line buckets must be positive and within the recognizer width limit"
        );
        buckets.sort_unstable();
        buckets.dedup();
        // This also prevents an accepted request from failing at a partial batch.
        buckets.retain(|&width| (1..=max_batch).all(|n| recognizer.supports_shape(n, width)));
        anyhow::ensure!(!buckets.is_empty(), "no usable line recognition buckets");
        Ok(Self {
            recognizer: Mutex::new(recognizer),
            buckets,
            max_batch,
            max_images,
            max_bytes,
            max_pixels,
        })
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct LineRequest {
    image: Option<String>,
    images: Option<Vec<String>>,
    batch_size: Option<i32>,
}

pub(super) async fn info(State(state): State<AppState>) -> Result<Json<Value>, ApiError> {
    let worker = state
        .lines
        .ok_or_else(|| ApiError::unavailable("line recognition is not enabled"))?;
    let native = worker
        .recognizer
        .lock()
        .map_err(|_| ApiError::internal("line recognizer lock was poisoned"))?
        .info_json()
        .map_err(|e| ApiError::internal(e.to_string()))?;
    Ok(Json(json!({"recognizer": native, "recognition_height": 48,
        "recognition_buckets": worker.buckets, "max_batch_size": worker.max_batch,
        "max_images": worker.max_images, "max_total_image_bytes": worker.max_bytes,
        "max_total_image_pixels": worker.max_pixels,
        "max_normalized_width": worker.buckets.last(),
        "input_contract": "upright single-line crops; no detection or rotation",
        "character_policy": "all", "score_mode": "model"})))
}

pub(super) async fn recognize(
    State(state): State<AppState>,
    Json(payload): Json<LineRequest>,
) -> Result<Json<Value>, ApiError> {
    let worker = state
        .lines
        .ok_or_else(|| ApiError::unavailable("line recognition is not enabled"))?;
    let permit = timeout(state.queue_timeout, state.semaphore.acquire_owned())
        .await
        .map_err(|_| ApiError::unavailable("line recognition queue timeout"))?
        .map_err(|_| ApiError::unavailable("recognition worker is closed"))?;
    let result = tokio::task::spawn_blocking(move || {
        let _permit = permit;
        recognize_blocking(&worker, payload)
    })
    .await
    .map_err(|e| ApiError::internal(format!("line recognition task failed: {e}")))??;
    Ok(Json(result))
}

fn bucket_for_shape(width: u32, height: u32, buckets: &[i32]) -> Result<i32, ApiError> {
    if width == 0 || height == 0 {
        return Err(ApiError::bad_request("line dimensions must be positive"));
    }
    let natural_width = (u64::from(width) * 48).div_ceil(u64::from(height));
    buckets
        .iter()
        .copied()
        .find(|&b| b as u64 >= natural_width)
        .ok_or_else(|| {
            ApiError::bad_request(format!(
                "line normalized width {natural_width} exceeds maximum {}; split the line crop",
                buckets.last().copied().unwrap_or(0)
            ))
        })
}

fn write_line_tensor(image: &RgbImage, bucket: usize, out: &mut [f32]) -> Result<(), ApiError> {
    // Enforce the line contract before using the shared RGB resize primitive.
    // Unlike glyph containment, text always occupies height 48 and is never
    // squeezed to fit a narrower canvas. Zero is normalized right padding.
    bucket_for_shape(image.width(), image.height(), &[bucket as i32])?;
    if out.len() != 3 * 48 * bucket {
        return Err(ApiError::internal("line tensor length mismatch"));
    }
    out.fill(0.0);
    let layout = GlyphResizeLayout {
        resized_width: (u64::from(image.width()) * 48).div_ceil(u64::from(image.height())) as usize,
        resized_height: 48,
        y_offset: 0,
    };
    if image.width() as usize == layout.resized_width && image.height() == 48 {
        write_rgb_tensor(image, bucket, layout, out);
    } else {
        write_resized_rgb_tensor(image, bucket, layout, out);
    }
    Ok(())
}

fn decode_line_image(raw: &[u8], remaining_pixels: u64) -> Result<RgbImage, ApiError> {
    // Check dimensions before the decoder allocates pixel storage. The glyph
    // decoder's post-decode check alone is insufficient for aggregate budgets.
    let reader = image::ImageReader::new(std::io::Cursor::new(raw))
        .with_guessed_format()
        .map_err(|_| ApiError::bad_request("invalid line image header"))?;
    let (width, height) = reader
        .into_dimensions()
        .map_err(|_| ApiError::bad_request("base64 payload is not a readable image"))?;
    if u64::from(width) * u64::from(height) > remaining_pixels {
        return Err(ApiError::too_large(
            "line request exceeds total decoded pixel limit",
        ));
    }
    decode_image(raw, remaining_pixels)
}

fn recognize_blocking(worker: &LineWorker, payload: LineRequest) -> Result<Value, ApiError> {
    let started = Instant::now();
    let single = payload.image.is_some() && payload.images.is_none();
    let images = extract_images(payload.image, payload.images, worker.max_images)?;
    let batch = optional_int(
        payload.batch_size,
        worker.max_batch,
        1,
        worker.max_batch,
        "batch_size",
    )?;
    let mut remaining_bytes = worker.max_bytes;
    let mut remaining_pixels = worker.max_pixels;
    let count = images.len();
    let mut groups: BTreeMap<i32, Vec<(usize, RgbImage)>> = BTreeMap::new();
    // Decode and validate the entire logical request before submitting inference.
    // Retained RGB images are bounded by the aggregate pixel budget. Normalized
    // tensors are created only for one inference chunk at a time.
    for (index, encoded) in images.iter().enumerate() {
        let raw = decode_base64_image(encoded, remaining_bytes)?;
        remaining_bytes -= raw.len();
        let image = decode_line_image(&raw, remaining_pixels)?;
        remaining_pixels -= u64::from(image.width()) * u64::from(image.height());
        let bucket = bucket_for_shape(image.width(), image.height(), &worker.buckets)?;
        groups.entry(bucket).or_default().push((index, image));
    }
    let preprocess_ms = started.elapsed().as_secs_f64() * 1000.0;
    let mut predictions = vec![Value::Null; count];
    let mut chunks = Vec::new();
    let mut resize_ms = 0.0;
    let mut recognize_ms = 0.0;
    let mut native_predict_ms = 0.0;
    let mut recognizer = worker
        .recognizer
        .lock()
        .map_err(|_| ApiError::internal("line recognizer lock was poisoned"))?;
    for (bucket, images) in groups {
        let stride = 3 * 48 * bucket as usize;
        for chunk in images.chunks(batch as usize) {
            let resize_started = Instant::now();
            let mut tensor = vec![0.0; stride * chunk.len()];
            if chunk.len() > 1 && tensor.len() >= 131_072 {
                tensor
                    .par_chunks_mut(stride)
                    .zip(chunk.par_iter())
                    .try_for_each(|(out, (_, image))| {
                        write_line_tensor(image, bucket as usize, out)
                    })?;
            } else {
                for (row, (_, image)) in chunk.iter().enumerate() {
                    write_line_tensor(
                        image,
                        bucket as usize,
                        &mut tensor[row * stride..(row + 1) * stride],
                    )?;
                }
            }
            resize_ms += resize_started.elapsed().as_secs_f64() * 1000.0;
            let recognize_started = Instant::now();
            let response = recognizer
                .recognize_f32_json(
                    &tensor,
                    chunk.len() as i32,
                    bucket,
                    chunk.len() as i32,
                    false,
                    CharacterPolicy::All as i32,
                    ScoreMode::Model as i32,
                )
                .map_err(|e| ApiError::internal(e.to_string()))?;
            recognize_ms += recognize_started.elapsed().as_secs_f64() * 1000.0;
            native_predict_ms += response["elapsed_ms"].as_f64().unwrap_or(0.0);
            let values = response
                .get("predictions")
                .and_then(Value::as_array)
                .filter(|v| v.len() == chunk.len())
                .ok_or_else(|| ApiError::internal("native line result count mismatch"))?;
            for ((index, _), value) in chunk.iter().zip(values) {
                predictions[*index] = json!({
                    "index": index, "text": value["text"], "score": value["score"],
                    "class_ids": value["class_ids"], "per_char_scores": value["per_char_scores"],
                    "bucket_width": bucket,
                });
            }
            chunks.push(json!({"batch": chunk.len(), "width": bucket}));
        }
    }
    let mut result = json!({"count": count, "predictions": predictions,
        "recognition_chunks": chunks, "character_policy": "all", "score_mode": "model",
        "timing": {"decode_ms": preprocess_ms, "resize_ms": resize_ms,
            "recognize_ms": recognize_ms, "native_predict_ms": native_predict_ms,
            "total_ms": started.elapsed().as_secs_f64() * 1000.0}});
    if single {
        result["prediction"] = result["predictions"][0].clone();
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn line_geometry_and_overflow() {
        let buckets = [128, 640, 960, 1280, 1600, 2400, 3200];
        assert_eq!(bucket_for_shape(129, 48, &buckets).unwrap(), 640);
        assert_eq!(bucket_for_shape(6400, 96, &buckets).unwrap(), 3200);
        assert_eq!(bucket_for_shape(1, 48, &buckets).unwrap(), 128);
        assert!(bucket_for_shape(3201, 48, &buckets).is_err());
        assert!(bucket_for_shape(1, 0, &buckets).is_err());
    }
    #[test]
    fn line_tensor_preserves_rgb_and_padding() {
        let image = RgbImage::from_pixel(129, 48, image::Rgb([0, 127, 255]));
        let mut tensor = vec![9.0; 3 * 48 * 640];
        write_line_tensor(&image, 640, &mut tensor).unwrap();
        assert_eq!(tensor[0], -1.0);
        assert_eq!(tensor[2 * 48 * 640], 1.0);
        assert_eq!(tensor[128], -1.0);
        assert_eq!(tensor[129], 0.0);
        assert!(write_line_tensor(&image, 128, &mut tensor).is_err());
    }
    #[test]
    fn rejects_glyph_options() {
        assert!(serde_json::from_value::<LineRequest>(json!({"image":"x", "width":80})).is_err());
    }
}
