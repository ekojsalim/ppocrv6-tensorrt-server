mod native;

use std::collections::HashMap;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use axum::body::Body;
use axum::extract::{DefaultBodyLimit, State};
use axum::http::{Response, StatusCode};
use axum::response::{IntoResponse, Json};
use axum::routing::{get, post};
use axum::Router;
use base64::engine::general_purpose::STANDARD as BASE64;
use base64::Engine as _;
use clap::Parser;
use image::{DynamicImage, GenericImageView, RgbImage, RgbaImage};
use native::{FullPageNativeConfig, NativeConfig, NativeFullPage, NativeRecognizer};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use tokio::sync::Semaphore;
use tokio::time::timeout;

#[derive(Debug, Parser)]
#[command(about = "Native PP-OCRv6 TensorRT HTTP server")]
struct Args {
    #[arg(
        long,
        env = "PPOCRV6_NATIVE_LIB",
        default_value = "native/build-trt-opencv/libppocrv6_native_c_abi.so"
    )]
    native_lib: PathBuf,

    #[arg(
        long,
        env = "PPOCRV6_GLYPH_ENGINE",
        default_value = "artifacts/ppocrv6-medium/engines/rec-hidden-multiprofile-glyph-line.trt"
    )]
    engine: String,

    #[arg(
        long,
        env = "PPOCRV6_GLYPH_WEIGHT",
        default_value = "artifacts/ppocrv6-medium/classifier/weight.fp16.bin"
    )]
    weight: String,

    #[arg(
        long,
        env = "PPOCRV6_GLYPH_BIAS",
        default_value = "artifacts/ppocrv6-medium/classifier/bias.fp16.bin"
    )]
    bias: String,

    #[arg(
        long,
        env = "PPOCRV6_GLYPH_CHARACTERS",
        default_value = "artifacts/ppocrv6-medium/classifier/characters.txt"
    )]
    characters: String,

    #[arg(
        long,
        env = "PPOCRV6_GLYPH_CHARACTER_POLICY",
        default_value = "cjk_focus_fallback"
    )]
    glyph_character_policy: String,

    #[arg(long, env = "PPOCRV6_GLYPH_HOST", default_value = "127.0.0.1")]
    host: String,

    #[arg(long, env = "PPOCRV6_GLYPH_PORT", default_value_t = 8184)]
    port: u16,

    #[arg(long, env = "PPOCRV6_GLYPH_WIDTH", default_value_t = 80)]
    width: i32,

    #[arg(long, env = "PPOCRV6_GLYPH_BATCH_SIZE", default_value_t = 256)]
    batch_size: i32,

    #[arg(long, env = "PPOCRV6_GLYPH_MAX_WIDTH", default_value_t = 128)]
    max_width: i32,

    #[arg(long, env = "PPOCRV6_GLYPH_MAX_BATCH_SIZE", default_value_t = 256)]
    max_batch_size: i32,

    #[arg(long, env = "PPOCRV6_GLYPH_MAX_IMAGES", default_value_t = 1024)]
    max_images: usize,

    #[arg(long, env = "PPOCRV6_GLYPH_MAX_REQUEST_BYTES", default_value_t = 16 * 1024 * 1024)]
    max_request_bytes: usize,

    #[arg(long, env = "PPOCRV6_GLYPH_MAX_IMAGE_BYTES", default_value_t = 1024 * 1024)]
    max_image_bytes: usize,

    #[arg(long, env = "PPOCRV6_GLYPH_MAX_IMAGE_PIXELS", default_value_t = 1024 * 1024)]
    max_image_pixels: u64,

    #[arg(long, env = "PPOCRV6_GLYPH_VOCAB_SIZE", default_value_t = 18710)]
    vocab_size: i32,

    #[arg(long, env = "PPOCRV6_GLYPH_HIDDEN_SIZE", default_value_t = 192)]
    hidden_size: i32,

    #[arg(long, env = "PPOCRV6_GLYPH_VOCAB_TILE_SIZE", default_value_t = 512)]
    vocab_tile_size: i32,

    #[arg(long, env = "PPOCRV6_GLYPH_BLANK_ID", default_value_t = 0)]
    blank_id: i32,

    #[arg(long, env = "PPOCRV6_GLYPH_WARMUP_RUNS", default_value_t = 1)]
    warmup_runs: i32,

    #[arg(long, env = "PPOCRV6_GLYPH_WORKER_PERMITS", default_value_t = 1)]
    worker_permits: usize,

    #[arg(long, env = "PPOCRV6_GLYPH_QUEUE_TIMEOUT_MS", default_value_t = 30_000)]
    queue_timeout_ms: u64,

    #[arg(long, env = "PPOCRV6_OCR_ENABLE", default_value_t = false)]
    enable_ocr: bool,

    #[arg(
        long,
        env = "PPOCRV6_OCR_DET_ENGINE",
        default_value = "artifacts/ppocrv6-medium/engines/det-fp16-b1-h256-1280-w256-1280.trt"
    )]
    ocr_det_engine: String,

    #[arg(long, env = "PPOCRV6_OCR_DET_DEFAULT_HEIGHT", default_value_t = 1280)]
    ocr_det_default_height: i32,

    #[arg(long, env = "PPOCRV6_OCR_DET_DEFAULT_WIDTH", default_value_t = 992)]
    ocr_det_default_width: i32,

    #[arg(long, env = "PPOCRV6_OCR_DET_MAX_HEIGHT", default_value_t = 1280)]
    ocr_det_max_height: i32,

    #[arg(long, env = "PPOCRV6_OCR_DET_MAX_WIDTH", default_value_t = 1280)]
    ocr_det_max_width: i32,

    #[arg(long, env = "PPOCRV6_OCR_DET_WARMUP_RUNS", default_value_t = 0)]
    ocr_det_warmup_runs: i32,

    #[arg(
        long,
        env = "PPOCRV6_OCR_REC_ENGINE",
        default_value = "artifacts/ppocrv6-medium/engines/rec-hidden-multiprofile-glyph-line.trt"
    )]
    ocr_rec_engine: String,

    #[arg(long, env = "PPOCRV6_OCR_REC_DEFAULT_WIDTH", default_value_t = 1600)]
    ocr_rec_default_width: i32,

    #[arg(long, env = "PPOCRV6_OCR_REC_BATCH_SIZE", default_value_t = 8)]
    ocr_rec_batch_size: i32,

    #[arg(long, env = "PPOCRV6_OCR_REC_MAX_BATCH_SIZE", default_value_t = 12)]
    ocr_rec_max_batch_size: i32,

    #[arg(long, env = "PPOCRV6_OCR_REC_MAX_WIDTH", default_value_t = 3200)]
    ocr_rec_max_width: i32,

    #[arg(long, env = "PPOCRV6_OCR_REC_WARMUP_RUNS", default_value_t = 0)]
    ocr_rec_warmup_runs: i32,

    #[arg(
        long,
        env = "PPOCRV6_OCR_REC_BUCKETS",
        default_value = "128,256,384,512,640,960,1280,1600,2400,3200"
    )]
    ocr_rec_buckets: String,

    #[arg(long, env = "PPOCRV6_OCR_DET_LIMIT_SIDE_LEN", default_value_t = 1280)]
    ocr_det_limit_side_len: u32,

    #[arg(long, env = "PPOCRV6_OCR_DET_LIMIT_TYPE", default_value = "max")]
    ocr_det_limit_type: String,

    #[arg(long, env = "PPOCRV6_OCR_DET_MAX_SIDE_LIMIT", default_value_t = 4000)]
    ocr_det_max_side_limit: u32,

    #[arg(long, env = "PPOCRV6_OCR_DET_THRESH", default_value_t = 0.2)]
    ocr_det_thresh: f32,

    #[arg(long, env = "PPOCRV6_OCR_DET_BOX_THRESH", default_value_t = 0.45)]
    ocr_det_box_thresh: f32,

    #[arg(long, env = "PPOCRV6_OCR_DET_UNCLIP_RATIO", default_value_t = 1.4)]
    ocr_det_unclip_ratio: f32,

    #[arg(long, env = "PPOCRV6_OCR_DET_MAX_CANDIDATES", default_value_t = 3000)]
    ocr_det_max_candidates: i32,

    #[arg(long, env = "PPOCRV6_OCR_DET_MIN_SIZE", default_value_t = 3)]
    ocr_det_min_size: i32,

    #[arg(long, env = "PPOCRV6_OCR_MAX_IMAGE_BYTES", default_value_t = 16 * 1024 * 1024)]
    ocr_max_image_bytes: usize,

    #[arg(
        long,
        env = "PPOCRV6_OCR_MAX_IMAGE_PIXELS",
        default_value_t = 16_000_000
    )]
    ocr_max_image_pixels: u64,
}

#[derive(Clone)]
struct AppState {
    recognizer: Arc<Mutex<NativeRecognizer>>,
    ocr: Option<Arc<Mutex<NativeFullPage>>>,
    semaphore: Arc<Semaphore>,
    limits: Limits,
    ocr_limits: OcrLimits,
    defaults: Defaults,
    queue_timeout: Duration,
}

#[derive(Debug, Clone, Serialize)]
struct Limits {
    max_request_bytes: usize,
    max_images: usize,
    max_width: i32,
    max_batch_size: i32,
    max_image_bytes: usize,
    max_image_pixels: u64,
}

#[derive(Debug, Clone)]
struct Defaults {
    width: i32,
    batch_size: i32,
    character_policy: CharacterPolicy,
}

#[derive(Debug, Clone, Serialize)]
struct OcrLimits {
    max_image_bytes: usize,
    max_image_pixels: u64,
    detector_limit_side_len: u32,
    detector_limit_type: String,
    detector_max_side_limit: u32,
}

#[derive(Debug, Deserialize)]
struct RecognizeRequest {
    image: Option<String>,
    images: Option<Vec<String>>,
    width: Option<i32>,
    batch_size: Option<i32>,
    return_timesteps: Option<bool>,
    character_policy: Option<String>,
}

#[derive(Debug, Deserialize)]
struct OcrRecognizeRequest {
    image: String,
}

#[derive(Debug)]
struct ApiError {
    status: StatusCode,
    message: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
enum CharacterPolicy {
    All = 0,
    SuppressAscii = 1,
    CjkFocus = 2,
    CjkFocusFallback = 3,
}

impl CharacterPolicy {
    fn parse(value: &str) -> Option<Self> {
        match value {
            "all" => Some(Self::All),
            "suppress_ascii" | "suppress-ascii" => Some(Self::SuppressAscii),
            "cjk_focus" | "cjk-focus" => Some(Self::CjkFocus),
            "cjk_focus_fallback" | "cjk-focus-fallback" => Some(Self::CjkFocusFallback),
            _ => None,
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::All => "all",
            Self::SuppressAscii => "suppress_ascii",
            Self::CjkFocus => "cjk_focus",
            Self::CjkFocusFallback => "cjk_focus_fallback",
        }
    }
}

impl ApiError {
    fn bad_request(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            message: message.into(),
        }
    }

    fn too_large(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::PAYLOAD_TOO_LARGE,
            message: message.into(),
        }
    }

    fn unavailable(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::SERVICE_UNAVAILABLE,
            message: message.into(),
        }
    }

    fn internal(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            message: message.into(),
        }
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response<Body> {
        let body = Json(json!({
            "error": self.message,
            "status": self.status.as_u16(),
        }));
        (self.status, body).into_response()
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    validate_args(&args)?;
    let default_character_policy = CharacterPolicy::parse(&args.glyph_character_policy)
        .expect("glyph character policy was validated");

    let native_config = NativeConfig {
        engine_path: &args.engine,
        weight_path: &args.weight,
        bias_path: &args.bias,
        characters_path: &args.characters,
        default_width: args.width,
        default_batch_size: args.batch_size,
        max_batch_size: args.max_batch_size,
        max_width: args.max_width,
        vocab_size: args.vocab_size,
        hidden_size: args.hidden_size,
        vocab_tile_size: args.vocab_tile_size,
        blank_id: args.blank_id,
        warmup_runs: args.warmup_runs,
    };
    let recognizer = NativeRecognizer::load(&args.native_lib, &native_config)?;
    validate_limit_type_config(&args.ocr_det_limit_type)?;
    let ocr = if args.enable_ocr {
        let full_page_config = FullPageNativeConfig {
            detector_engine_path: &args.ocr_det_engine,
            detector_default_batch: 1,
            detector_default_height: args.ocr_det_default_height,
            detector_default_width: args.ocr_det_default_width,
            detector_max_batch: 1,
            detector_max_height: args.ocr_det_max_height,
            detector_max_width: args.ocr_det_max_width,
            detector_warmup_runs: args.ocr_det_warmup_runs,
            detector_limit_side_len: args.ocr_det_limit_side_len as i32,
            detector_limit_type: &args.ocr_det_limit_type,
            detector_max_side_limit: args.ocr_det_max_side_limit as i32,
            recognizer_engine_path: &args.ocr_rec_engine,
            weight_path: &args.weight,
            bias_path: &args.bias,
            characters_path: &args.characters,
            recognizer_default_width: args.ocr_rec_default_width,
            recognizer_default_batch_size: args.ocr_rec_batch_size,
            recognizer_max_batch_size: args.ocr_rec_max_batch_size,
            recognizer_max_width: args.ocr_rec_max_width,
            vocab_size: args.vocab_size,
            hidden_size: args.hidden_size,
            vocab_tile_size: args.vocab_tile_size,
            blank_id: args.blank_id,
            recognizer_warmup_runs: args.ocr_rec_warmup_runs,
            det_thresh: args.ocr_det_thresh,
            det_box_thresh: args.ocr_det_box_thresh,
            det_unclip_ratio: args.ocr_det_unclip_ratio,
            det_max_candidates: args.ocr_det_max_candidates,
            det_min_size: args.ocr_det_min_size,
            recognition_height: 48,
            recognition_buckets_csv: &args.ocr_rec_buckets,
        };
        Some(Arc::new(Mutex::new(NativeFullPage::load(
            &args.native_lib,
            &full_page_config,
        )?)))
    } else {
        None
    };
    let state = AppState {
        recognizer: Arc::new(Mutex::new(recognizer)),
        ocr,
        semaphore: Arc::new(Semaphore::new(args.worker_permits)),
        limits: Limits {
            max_request_bytes: args.max_request_bytes,
            max_images: args.max_images,
            max_width: args.max_width,
            max_batch_size: args.max_batch_size,
            max_image_bytes: args.max_image_bytes,
            max_image_pixels: args.max_image_pixels,
        },
        ocr_limits: OcrLimits {
            max_image_bytes: args.ocr_max_image_bytes,
            max_image_pixels: args.ocr_max_image_pixels,
            detector_limit_side_len: args.ocr_det_limit_side_len,
            detector_limit_type: args.ocr_det_limit_type.clone(),
            detector_max_side_limit: args.ocr_det_max_side_limit,
        },
        defaults: Defaults {
            width: args.width,
            batch_size: args.batch_size,
            character_policy: default_character_policy,
        },
        queue_timeout: Duration::from_millis(args.queue_timeout_ms),
    };
    let ocr_body_limit = args
        .ocr_max_image_bytes
        .saturating_mul(4)
        .saturating_div(3)
        .saturating_add(4096);
    let request_body_limit = if args.enable_ocr {
        args.max_request_bytes.max(ocr_body_limit)
    } else {
        args.max_request_bytes
    };

    let app = Router::new()
        .route("/health", get(health).options(options_ok))
        .route("/healthz", get(health).options(options_ok))
        .route("/v1/glyphs/info", get(info).options(options_ok))
        .route("/v1/glyphs/recognize", post(recognize).options(options_ok))
        .route("/v1/ocr/info", get(ocr_info).options(options_ok))
        .route("/v1/ocr/recognize", post(ocr_recognize).options(options_ok))
        .route(
            "/v1/pages/recognize",
            post(ocr_recognize).options(options_ok),
        )
        .layer(DefaultBodyLimit::max(request_body_limit))
        .with_state(state);

    let addr: SocketAddr = format!("{}:{}", args.host, args.port).parse()?;
    let listener = tokio::net::TcpListener::bind(addr).await?;
    println!(
        "{{\"event\":\"startup\",\"listening\":\"http://{}\",\"glyph_recognize\":\"http://{}/v1/glyphs/recognize\",\"ocr_enabled\":{},\"ocr_recognize\":\"http://{}/v1/ocr/recognize\"}}",
        addr, addr, args.enable_ocr, addr
    );
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await?;
    Ok(())
}

async fn shutdown_signal() {
    let ctrl_c = async {
        if let Err(err) = tokio::signal::ctrl_c().await {
            eprintln!("failed to install Ctrl-C shutdown handler: {err}");
        }
    };

    #[cfg(unix)]
    let terminate = async {
        match tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()) {
            Ok(mut signal) => {
                signal.recv().await;
            }
            Err(err) => {
                eprintln!("failed to install SIGTERM shutdown handler: {err}");
                std::future::pending::<()>().await;
            }
        }
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => {},
        _ = terminate => {},
    }
}

fn validate_args(args: &Args) -> anyhow::Result<()> {
    if args.width <= 0 || args.batch_size <= 0 {
        anyhow::bail!("--width and --batch-size must be positive");
    }
    if args.max_width <= 0 || args.max_batch_size <= 0 || args.max_images == 0 {
        anyhow::bail!("max width, max batch size, and max images must be positive");
    }
    if args.width > args.max_width || args.batch_size > args.max_batch_size {
        anyhow::bail!("default width/batch-size must not exceed max limits");
    }
    if args.worker_permits == 0 {
        anyhow::bail!("--worker-permits must be positive");
    }
    if CharacterPolicy::parse(&args.glyph_character_policy).is_none() {
        anyhow::bail!(
            "--glyph-character-policy must be one of cjk_focus, cjk_focus_fallback, suppress_ascii, all"
        );
    }
    validate_limit_type_config(&args.ocr_det_limit_type)?;
    if args.ocr_det_default_height <= 0
        || args.ocr_det_default_width <= 0
        || args.ocr_det_max_height <= 0
        || args.ocr_det_max_width <= 0
    {
        anyhow::bail!("OCR detector dimensions must be positive");
    }
    if args.ocr_det_default_height > args.ocr_det_max_height
        || args.ocr_det_default_width > args.ocr_det_max_width
    {
        anyhow::bail!("OCR detector defaults must not exceed max detector dimensions");
    }
    if args.ocr_rec_default_width <= 0
        || args.ocr_rec_batch_size <= 0
        || args.ocr_rec_max_batch_size <= 0
        || args.ocr_rec_max_width <= 0
    {
        anyhow::bail!("OCR recognizer dimensions and batch sizes must be positive");
    }
    if args.ocr_rec_default_width > args.ocr_rec_max_width
        || args.ocr_rec_batch_size > args.ocr_rec_max_batch_size
    {
        anyhow::bail!("OCR recognizer defaults must not exceed max recognizer limits");
    }
    if args.ocr_det_limit_side_len == 0 || args.ocr_det_max_side_limit == 0 {
        anyhow::bail!("OCR detector resize limits must be positive");
    }
    if args.ocr_det_max_candidates <= 0 || args.ocr_det_min_size <= 0 {
        anyhow::bail!("OCR detector postprocess limits must be positive");
    }
    if args.ocr_max_image_bytes == 0 || args.ocr_max_image_pixels == 0 {
        anyhow::bail!("OCR image limits must be positive");
    }
    Ok(())
}

fn validate_limit_type_config(value: &str) -> anyhow::Result<()> {
    match value {
        "max" | "min" | "resize_long" => Ok(()),
        _ => anyhow::bail!("--ocr-det-limit-type must be one of max, min, resize_long"),
    }
}

async fn options_ok() -> StatusCode {
    StatusCode::NO_CONTENT
}

async fn health() -> Json<Value> {
    Json(json!({ "ok": true }))
}

async fn info(State(state): State<AppState>) -> Result<Json<Value>, ApiError> {
    let mut recognizer = state
        .recognizer
        .lock()
        .map_err(|_| ApiError::internal("native recognizer lock was poisoned"))?;
    let mut info = recognizer
        .info_json()
        .map_err(|err| ApiError::internal(err.to_string()))?;
    if let Value::Object(map) = &mut info {
        map.insert("limits".to_string(), json!(state.limits));
        map.insert(
            "default_character_policy".to_string(),
            json!(state.defaults.character_policy.as_str()),
        );
        map.insert(
            "score_type".to_string(),
            json!(match state.defaults.character_policy {
                CharacterPolicy::All => "probability",
                CharacterPolicy::SuppressAscii
                | CharacterPolicy::CjkFocus
                | CharacterPolicy::CjkFocusFallback => {
                    "conditional_probability"
                }
            }),
        );
        map.insert(
            "score_types_by_character_policy".to_string(),
            json!({
                "cjk_focus": "conditional_probability",
                "cjk_focus_fallback": "conditional_probability",
                "suppress_ascii": "conditional_probability",
                "all": "probability",
            }),
        );
    }
    Ok(Json(info))
}

async fn ocr_info(State(state): State<AppState>) -> Result<Json<Value>, ApiError> {
    let ocr = state
        .ocr
        .as_ref()
        .ok_or_else(|| ApiError::unavailable("full-page OCR worker is not enabled"))?;
    let mut worker = ocr
        .lock()
        .map_err(|_| ApiError::internal("native full-page OCR lock was poisoned"))?;
    let mut info = worker
        .info_json()
        .map_err(|err| ApiError::internal(err.to_string()))?;
    if let Value::Object(map) = &mut info {
        map.insert("ocr_limits".to_string(), json!(state.ocr_limits));
        map.insert(
            "input_contract".to_string(),
            json!({
                "http": "base64 PNG/JPEG/etc in JSON field 'image'",
                "native": "decoded RGB bytes; C++ owns detector preprocessing",
            }),
        );
    }
    Ok(Json(info))
}

async fn recognize(
    State(state): State<AppState>,
    Json(payload): Json<RecognizeRequest>,
) -> Result<Json<Value>, ApiError> {
    let permit = timeout(state.queue_timeout, state.semaphore.clone().acquire_owned())
        .await
        .map_err(|_| ApiError::unavailable("recognition queue timeout"))?
        .map_err(|_| ApiError::unavailable("recognition worker is closed"))?;

    let state_for_blocking = state.clone();
    let value = tokio::task::spawn_blocking(move || {
        let _permit = permit;
        recognize_blocking(state_for_blocking, payload)
    })
    .await
    .map_err(|err| ApiError::internal(format!("recognition task failed: {err}")))??;

    Ok(Json(value))
}

async fn ocr_recognize(
    State(state): State<AppState>,
    Json(payload): Json<OcrRecognizeRequest>,
) -> Result<Json<Value>, ApiError> {
    let permit = timeout(state.queue_timeout, state.semaphore.clone().acquire_owned())
        .await
        .map_err(|_| ApiError::unavailable("OCR queue timeout"))?
        .map_err(|_| ApiError::unavailable("OCR worker is closed"))?;

    let state_for_blocking = state.clone();
    let value = tokio::task::spawn_blocking(move || {
        let _permit = permit;
        ocr_recognize_blocking(state_for_blocking, payload)
    })
    .await
    .map_err(|err| ApiError::internal(format!("OCR task failed: {err}")))??;

    Ok(Json(value))
}

fn recognize_blocking(state: AppState, payload: RecognizeRequest) -> Result<Value, ApiError> {
    let request_started = Instant::now();
    let character_policy = match payload.character_policy.as_deref() {
        Some(value) => CharacterPolicy::parse(value).ok_or_else(|| {
            ApiError::bad_request(
                "'character_policy' must be one of 'cjk_focus', 'cjk_focus_fallback', 'suppress_ascii', or 'all'",
            )
        })?,
        None => state.defaults.character_policy,
    };
    let single_image_request = payload.image.is_some() && payload.images.is_none();
    let image_values = extract_images(payload.image, payload.images, state.limits.max_images)?;
    let width = optional_int(
        payload.width,
        state.defaults.width,
        1,
        state.limits.max_width,
        "width",
    )?;
    let batch_size = optional_int(
        payload.batch_size,
        state.defaults.batch_size,
        1,
        state.limits.max_batch_size,
        "batch_size",
    )?;
    let return_timesteps = payload.return_timesteps.unwrap_or(false);
    let preprocess_started = Instant::now();
    let preprocess = preprocess_images(
        &image_values,
        width,
        state.limits.max_image_bytes,
        state.limits.max_image_pixels,
    )?;
    let preprocess_ms = preprocess_started.elapsed().as_secs_f64() * 1000.0;

    let mut recognizer = state
        .recognizer
        .lock()
        .map_err(|_| ApiError::internal("native recognizer lock was poisoned"))?;
    let mut response = recognizer
        .recognize_f32_json(
            &preprocess.tensor,
            image_values.len() as i32,
            width,
            batch_size,
            return_timesteps,
            character_policy as i32,
        )
        .map_err(|err| ApiError::internal(err.to_string()))?;
    let blocking_ms = request_started.elapsed().as_secs_f64() * 1000.0;
    attach_rust_timings(&mut response, &preprocess.stats, preprocess_ms, blocking_ms);

    if single_image_request {
        if let Value::Object(map) = &mut response {
            if let Some(prediction) = map
                .get("predictions")
                .and_then(Value::as_array)
                .and_then(|items| items.first())
                .cloned()
            {
                map.insert("prediction".to_string(), prediction);
            }
        }
    }
    Ok(response)
}

fn ocr_recognize_blocking(
    state: AppState,
    payload: OcrRecognizeRequest,
) -> Result<Value, ApiError> {
    if payload.image.is_empty() {
        return Err(ApiError::bad_request("'image' must not be empty"));
    }
    let worker = state
        .ocr
        .ok_or_else(|| ApiError::unavailable("full-page OCR worker is not enabled"))?;
    let request_started = Instant::now();

    let preprocess_started = Instant::now();
    let preprocess = preprocess_ocr_image(
        &payload.image,
        state.ocr_limits.max_image_bytes,
        state.ocr_limits.max_image_pixels,
    )?;
    let preprocess_ms = preprocess_started.elapsed().as_secs_f64() * 1000.0;

    let mut worker = worker
        .lock()
        .map_err(|_| ApiError::internal("native full-page OCR lock was poisoned"))?;
    let mut response = worker
        .recognize_image_json(
            preprocess.image.as_raw(),
            preprocess.image.height() as i32,
            preprocess.image.width() as i32,
            (preprocess.image.width() * 3) as i32,
        )
        .map_err(|err| ApiError::internal(err.to_string()))?;
    let blocking_ms = request_started.elapsed().as_secs_f64() * 1000.0;
    attach_ocr_rust_timings(&mut response, &preprocess, preprocess_ms, blocking_ms);
    Ok(response)
}

#[derive(Debug, Default)]
struct PreprocessStats {
    base64_decode_ms: f64,
    image_decode_ms: f64,
    resize_normalize_ms: f64,
    cache_hits: usize,
}

impl PreprocessStats {
    fn add(&mut self, other: &PreprocessStats) {
        self.base64_decode_ms += other.base64_decode_ms;
        self.image_decode_ms += other.image_decode_ms;
        self.resize_normalize_ms += other.resize_normalize_ms;
        self.cache_hits += other.cache_hits;
    }
}

struct PreprocessOutput {
    tensor: Vec<f32>,
    stats: PreprocessStats,
}

#[derive(Debug, Default)]
struct OcrPreprocessStats {
    base64_decode_ms: f64,
    image_decode_ms: f64,
}

struct OcrPreprocessOutput {
    image: RgbImage,
    stats: OcrPreprocessStats,
}

fn preprocess_ocr_image(
    value: &str,
    max_image_bytes: usize,
    max_image_pixels: u64,
) -> Result<OcrPreprocessOutput, ApiError> {
    let mut stats = OcrPreprocessStats::default();

    let started = Instant::now();
    let raw = decode_base64_image(value, max_image_bytes)?;
    stats.base64_decode_ms += started.elapsed().as_secs_f64() * 1000.0;

    let started = Instant::now();
    let image = decode_image(&raw, max_image_pixels)?;
    stats.image_decode_ms += started.elapsed().as_secs_f64() * 1000.0;

    Ok(OcrPreprocessOutput { image, stats })
}

fn extract_images(
    image: Option<String>,
    images: Option<Vec<String>>,
    max_images: usize,
) -> Result<Vec<String>, ApiError> {
    match (image, images) {
        (Some(_), Some(_)) => Err(ApiError::bad_request(
            "request must include only one of 'image' or 'images'",
        )),
        (None, None) => Err(ApiError::bad_request(
            "request must include 'image' or 'images'",
        )),
        (Some(value), None) => {
            if value.is_empty() {
                return Err(ApiError::bad_request("'image' must not be empty"));
            }
            Ok(vec![value])
        }
        (None, Some(values)) => {
            if values.is_empty() {
                return Err(ApiError::bad_request("'images' must be a non-empty list"));
            }
            if values.len() > max_images {
                return Err(ApiError::too_large(format!(
                    "too many images: {} > {}",
                    values.len(),
                    max_images
                )));
            }
            if values.iter().any(String::is_empty) {
                return Err(ApiError::bad_request("'images' values must not be empty"));
            }
            Ok(values)
        }
    }
}

fn optional_int(
    value: Option<i32>,
    default: i32,
    min_value: i32,
    max_value: i32,
    key: &str,
) -> Result<i32, ApiError> {
    let value = value.unwrap_or(default);
    if value < min_value || value > max_value {
        return Err(ApiError::bad_request(format!(
            "'{key}' must be between {min_value} and {max_value}"
        )));
    }
    Ok(value)
}

fn preprocess_images(
    image_values: &[String],
    width: i32,
    max_image_bytes: usize,
    max_image_pixels: u64,
) -> Result<PreprocessOutput, ApiError> {
    let width_usize = width as usize;
    let image_stride = 3 * 48 * width_usize;
    let mut tensor = vec![0.0_f32; image_values.len() * image_stride];
    let mut stats = PreprocessStats::default();
    let mut first_by_payload: HashMap<&str, usize> = HashMap::new();
    let mut unique_jobs: Vec<(usize, &str)> = Vec::new();
    let mut copy_jobs: Vec<(usize, usize)> = Vec::new();

    for (index, value) in image_values.iter().enumerate() {
        if let Some(&cached_index) = first_by_payload.get(value.as_str()) {
            copy_jobs.push((index, cached_index));
        } else {
            first_by_payload.insert(value.as_str(), index);
            unique_jobs.push((index, value.as_str()));
        }
    }

    let results = if unique_jobs.len() > 1 {
        unique_jobs
            .par_iter()
            .map(|(index, value)| {
                preprocess_one_image(
                    *index,
                    value,
                    width_usize,
                    image_stride,
                    max_image_bytes,
                    max_image_pixels,
                )
            })
            .collect::<Vec<_>>()
    } else {
        unique_jobs
            .iter()
            .map(|(index, value)| {
                preprocess_one_image(
                    *index,
                    value,
                    width_usize,
                    image_stride,
                    max_image_bytes,
                    max_image_pixels,
                )
            })
            .collect::<Vec<_>>()
    };

    for result in results {
        let (index, data, item_stats) = result?;
        let start = index * image_stride;
        let end = start + image_stride;
        tensor[start..end].copy_from_slice(&data);
        stats.add(&item_stats);
    }

    for (index, cached_index) in copy_jobs {
        let start = index * image_stride;
        let cached_start = cached_index * image_stride;
        let cached_end = cached_start + image_stride;
        tensor.copy_within(cached_start..cached_end, start);
        stats.cache_hits += 1;
    }
    Ok(PreprocessOutput { tensor, stats })
}

fn preprocess_one_image(
    index: usize,
    value: &str,
    width: usize,
    image_stride: usize,
    max_image_bytes: usize,
    max_image_pixels: u64,
) -> Result<(usize, Vec<f32>, PreprocessStats), ApiError> {
    let mut stats = PreprocessStats::default();
    let mut tensor = vec![0.0_f32; image_stride];

    let started = Instant::now();
    let raw = decode_base64_image(value, max_image_bytes)?;
    stats.base64_decode_ms += started.elapsed().as_secs_f64() * 1000.0;

    let started = Instant::now();
    let image = decode_image(&raw, max_image_pixels)?;
    stats.image_decode_ms += started.elapsed().as_secs_f64() * 1000.0;

    let started = Instant::now();
    write_preprocessed_image(&image, width, &mut tensor)?;
    stats.resize_normalize_ms += started.elapsed().as_secs_f64() * 1000.0;

    Ok((index, tensor, stats))
}

fn decode_base64_image(value: &str, max_image_bytes: usize) -> Result<Vec<u8>, ApiError> {
    decode_base64_bytes(
        value,
        max_image_bytes,
        "invalid base64 image data",
        "image payload",
    )
}

fn decode_base64_bytes(
    value: &str,
    max_bytes: usize,
    invalid_message: &'static str,
    payload_name: &'static str,
) -> Result<Vec<u8>, ApiError> {
    let encoded = if starts_with_data_uri(value) {
        value.split_once(',').map(|(_, tail)| tail).unwrap_or(value)
    } else {
        value
    };
    let raw = BASE64
        .decode(encoded)
        .map_err(|_| ApiError::bad_request(invalid_message))?;
    if raw.len() > max_bytes {
        return Err(ApiError::too_large(format!(
            "{payload_name} is too large: {} > {} bytes",
            raw.len(),
            max_bytes
        )));
    }
    Ok(raw)
}

fn decode_image(raw: &[u8], max_image_pixels: u64) -> Result<RgbImage, ApiError> {
    if let Some(image) = decode_ppm_p6(raw, max_image_pixels)? {
        return Ok(image);
    }

    let image = image::load_from_memory(raw)
        .map_err(|_| ApiError::bad_request("base64 payload is not a readable image"))?;
    let (width, height) = image.dimensions();
    if width == 0 || height == 0 {
        return Err(ApiError::bad_request("image has invalid dimensions"));
    }
    let pixels = u64::from(width) * u64::from(height);
    if pixels > max_image_pixels {
        return Err(ApiError::too_large(format!(
            "image has too many pixels: {} > {}",
            pixels, max_image_pixels
        )));
    }
    Ok(to_rgb_on_white(image))
}

fn to_rgb_on_white(image: DynamicImage) -> RgbImage {
    match image {
        DynamicImage::ImageRgb8(rgb) => rgb,
        DynamicImage::ImageRgba8(rgba) => rgba_to_rgb_on_white(rgba),
        image => image.to_rgb8(),
    }
}

fn rgba_to_rgb_on_white(rgba: RgbaImage) -> RgbImage {
    let (width, height) = rgba.dimensions();
    let mut rgb = RgbImage::new(width, height);
    for y in 0..height {
        for x in 0..width {
            let pixel = rgba.get_pixel(x, y);
            let alpha = u16::from(pixel[3]);
            let inv_alpha = 255_u16 - alpha;
            let r = (u16::from(pixel[0]) * alpha + 255 * inv_alpha + 127) / 255;
            let g = (u16::from(pixel[1]) * alpha + 255 * inv_alpha + 127) / 255;
            let b = (u16::from(pixel[2]) * alpha + 255 * inv_alpha + 127) / 255;
            rgb.put_pixel(x, y, image::Rgb([r as u8, g as u8, b as u8]));
        }
    }
    rgb
}

fn write_preprocessed_image(
    image: &RgbImage,
    width: usize,
    out: &mut [f32],
) -> Result<(), ApiError> {
    let target_h = 48_u32;
    let image_w = image.width();
    let image_h = image.height();
    let resized_w =
        width.min((u64::from(target_h) * u64::from(image_w)).div_ceil(u64::from(image_h)) as usize);
    let resized_w = resized_w.max(1);
    let plane = 48 * width;
    if out.len() != 3 * plane {
        return Err(ApiError::internal("preprocess output slice has wrong size"));
    }
    if image.width() == resized_w as u32 && image.height() == target_h {
        write_rgb_tensor(image, width, resized_w, out);
        return Ok(());
    }
    write_resized_rgb_tensor(image, width, resized_w, out);
    Ok(())
}

fn write_rgb_tensor(image: &RgbImage, width: usize, resized_w: usize, out: &mut [f32]) {
    let plane = 48 * width;
    for y in 0..48_usize {
        for x in 0..resized_w {
            let pixel = image.get_pixel(x as u32, y as u32);
            for channel in 0..3 {
                out[channel * plane + y * width + x] =
                    (f32::from(pixel[channel]) / 255.0 - 0.5) / 0.5;
            }
        }
    }
}

fn write_resized_rgb_tensor(image: &RgbImage, width: usize, resized_w: usize, out: &mut [f32]) {
    let plane = 48 * width;
    let src_w = image.width() as usize;
    let src_h = image.height() as usize;
    let scale_x = src_w as f32 / resized_w as f32;
    let scale_y = src_h as f32 / 48.0;

    for y in 0..48_usize {
        let src_y = ((y as f32 + 0.5) * scale_y - 0.5).max(0.0);
        let y0 = (src_y.floor() as usize).min(src_h - 1);
        let y1 = (y0 + 1).min(src_h - 1);
        let wy = src_y - y0 as f32;

        for x in 0..resized_w {
            let src_x = ((x as f32 + 0.5) * scale_x - 0.5).max(0.0);
            let x0 = (src_x.floor() as usize).min(src_w - 1);
            let x1 = (x0 + 1).min(src_w - 1);
            let wx = src_x - x0 as f32;

            let p00 = image.get_pixel(x0 as u32, y0 as u32);
            let p10 = image.get_pixel(x1 as u32, y0 as u32);
            let p01 = image.get_pixel(x0 as u32, y1 as u32);
            let p11 = image.get_pixel(x1 as u32, y1 as u32);

            for channel in 0..3 {
                let top = f32::from(p00[channel]) * (1.0 - wx) + f32::from(p10[channel]) * wx;
                let bottom = f32::from(p01[channel]) * (1.0 - wx) + f32::from(p11[channel]) * wx;
                let value = top * (1.0 - wy) + bottom * wy;
                out[channel * plane + y * width + x] = value / 127.5 - 1.0;
            }
        }
    }
}

fn starts_with_data_uri(value: &str) -> bool {
    value.len() >= 5 && value.as_bytes()[..5].eq_ignore_ascii_case(b"data:")
}

fn decode_ppm_p6(raw: &[u8], max_image_pixels: u64) -> Result<Option<RgbImage>, ApiError> {
    if raw.len() < 3 || &raw[..2] != b"P6" {
        return Ok(None);
    }
    let mut parser = PpmParser { raw, pos: 2 };
    let width = parser.next_usize()?;
    let height = parser.next_usize()?;
    let max_value = parser.next_usize()?;
    if width == 0 || height == 0 {
        return Err(ApiError::bad_request("image has invalid dimensions"));
    }
    let pixels = width as u64 * height as u64;
    if pixels > max_image_pixels {
        return Err(ApiError::too_large(format!(
            "image has too many pixels: {} > {}",
            pixels, max_image_pixels
        )));
    }
    if max_value != 255 {
        return Err(ApiError::bad_request("PPM max value must be 255"));
    }
    parser.skip_single_raster_whitespace();
    let bytes = width
        .checked_mul(height)
        .and_then(|value| value.checked_mul(3))
        .ok_or_else(|| ApiError::too_large("PPM image is too large"))?;
    if raw.len().saturating_sub(parser.pos) != bytes {
        return Err(ApiError::bad_request(
            "PPM raster size does not match header",
        ));
    }
    RgbImage::from_raw(width as u32, height as u32, raw[parser.pos..].to_vec())
        .map(Some)
        .ok_or_else(|| ApiError::bad_request("PPM payload is not a readable image"))
}

struct PpmParser<'a> {
    raw: &'a [u8],
    pos: usize,
}

impl PpmParser<'_> {
    fn next_usize(&mut self) -> Result<usize, ApiError> {
        self.skip_ws_and_comments();
        let start = self.pos;
        while self.pos < self.raw.len() && self.raw[self.pos].is_ascii_digit() {
            self.pos += 1;
        }
        if self.pos == start {
            return Err(ApiError::bad_request("invalid PPM header"));
        }
        std::str::from_utf8(&self.raw[start..self.pos])
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .ok_or_else(|| ApiError::bad_request("invalid PPM header number"))
    }

    fn skip_ws_and_comments(&mut self) {
        loop {
            while self.pos < self.raw.len() && self.raw[self.pos].is_ascii_whitespace() {
                self.pos += 1;
            }
            if self.pos < self.raw.len() && self.raw[self.pos] == b'#' {
                while self.pos < self.raw.len() && self.raw[self.pos] != b'\n' {
                    self.pos += 1;
                }
                continue;
            }
            break;
        }
    }

    fn skip_single_raster_whitespace(&mut self) {
        if self.pos < self.raw.len() && self.raw[self.pos].is_ascii_whitespace() {
            self.pos += 1;
        }
    }
}

fn attach_rust_timings(
    response: &mut Value,
    stats: &PreprocessStats,
    preprocess_ms: f64,
    blocking_ms: f64,
) {
    let timing = json!({
        "rust_preprocess_ms": preprocess_ms,
        "rust_base64_decode_ms": stats.base64_decode_ms,
        "rust_image_decode_ms": stats.image_decode_ms,
        "rust_resize_normalize_ms": stats.resize_normalize_ms,
        "rust_cache_hits": stats.cache_hits,
        "rust_blocking_ms": blocking_ms,
    });
    if let Value::Object(map) = response {
        map.insert("rust_timings".to_string(), timing.clone());
        if let Some(Value::Object(meta)) = map.get_mut("meta") {
            meta.insert("rust_preprocess_ms".to_string(), json!(preprocess_ms));
            meta.insert(
                "rust_base64_decode_ms".to_string(),
                json!(stats.base64_decode_ms),
            );
            meta.insert(
                "rust_image_decode_ms".to_string(),
                json!(stats.image_decode_ms),
            );
            meta.insert(
                "rust_resize_normalize_ms".to_string(),
                json!(stats.resize_normalize_ms),
            );
            meta.insert("rust_cache_hits".to_string(), json!(stats.cache_hits));
            meta.insert("rust_blocking_ms".to_string(), json!(blocking_ms));
        }
    }
}

fn attach_ocr_rust_timings(
    response: &mut Value,
    preprocess: &OcrPreprocessOutput,
    preprocess_ms: f64,
    blocking_ms: f64,
) {
    let timing = json!({
        "rust_preprocess_ms": preprocess_ms,
        "rust_base64_decode_ms": preprocess.stats.base64_decode_ms,
        "rust_image_decode_ms": preprocess.stats.image_decode_ms,
        "rust_blocking_ms": blocking_ms,
        "rust_source_shape": [preprocess.image.height(), preprocess.image.width()],
    });
    if let Value::Object(map) = response {
        map.insert("rust_timings".to_string(), timing.clone());
        if let Some(Value::Object(native_timing)) = map.get_mut("timing") {
            native_timing.insert("rust_preprocess_ms".to_string(), json!(preprocess_ms));
            native_timing.insert(
                "rust_base64_decode_ms".to_string(),
                json!(preprocess.stats.base64_decode_ms),
            );
            native_timing.insert(
                "rust_image_decode_ms".to_string(),
                json!(preprocess.stats.image_decode_ms),
            );
            native_timing.insert("rust_blocking_ms".to_string(), json!(blocking_ms));
        }
    }
}

#[cfg(test)]
mod tests {
    use clap::Parser;

    use super::{Args, CharacterPolicy};

    #[test]
    fn defaults_to_cjk_focus_fallback() {
        let args = Args::try_parse_from(["ppocrv6-tensorrt-server"]).unwrap();
        assert_eq!(args.glyph_character_policy, "cjk_focus_fallback");
    }

    #[test]
    fn parses_character_policies() {
        assert_eq!(
            CharacterPolicy::parse("cjk_focus"),
            Some(CharacterPolicy::CjkFocus)
        );
        assert_eq!(
            CharacterPolicy::parse("cjk-focus"),
            Some(CharacterPolicy::CjkFocus)
        );
        assert_eq!(
            CharacterPolicy::parse("cjk_focus_fallback"),
            Some(CharacterPolicy::CjkFocusFallback)
        );
        assert_eq!(
            CharacterPolicy::parse("cjk-focus-fallback"),
            Some(CharacterPolicy::CjkFocusFallback)
        );
        assert_eq!(
            CharacterPolicy::parse("suppress_ascii"),
            Some(CharacterPolicy::SuppressAscii)
        );
        assert_eq!(
            CharacterPolicy::parse("suppress-ascii"),
            Some(CharacterPolicy::SuppressAscii)
        );
        assert_eq!(CharacterPolicy::parse("all"), Some(CharacterPolicy::All));
        assert_eq!(CharacterPolicy::parse("cjk"), None);
    }
}
