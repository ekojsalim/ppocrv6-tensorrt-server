use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_float, c_int, c_uchar, c_void};
use std::path::Path;
use std::ptr;

use anyhow::{anyhow, Context, Result};
use libloading::Library;
use serde::Serialize;
use serde_json::Value;

type CreateFn = unsafe extern "C" fn(*const c_char, *mut *mut c_void, *mut *mut c_char) -> c_int;
type DestroyFn = unsafe extern "C" fn(*mut c_void);
type InfoFn = unsafe extern "C" fn(*mut c_void, *mut *mut c_char, *mut *mut c_char) -> c_int;
type RecognizeF32Fn = unsafe extern "C" fn(
    *mut c_void,
    *const c_float,
    c_int,
    c_int,
    c_int,
    c_int,
    *mut *mut c_char,
    *mut *mut c_char,
) -> c_int;
type FreeStringFn = unsafe extern "C" fn(*mut c_char);
type FullPageCreateFn =
    unsafe extern "C" fn(*const c_char, *mut *mut c_void, *mut *mut c_char) -> c_int;
type FullPageDestroyFn = unsafe extern "C" fn(*mut c_void);
type FullPageInfoFn =
    unsafe extern "C" fn(*mut c_void, *mut *mut c_char, *mut *mut c_char) -> c_int;
type FullPageRecognizeImageFn = unsafe extern "C" fn(
    *mut c_void,
    *const c_uchar,
    c_int,
    c_int,
    c_int,
    c_int,
    *mut *mut c_char,
    *mut *mut c_char,
) -> c_int;

#[derive(Debug, Serialize)]
pub struct NativeConfig<'a> {
    pub engine_path: &'a str,
    pub weight_path: &'a str,
    pub bias_path: &'a str,
    pub characters_path: &'a str,
    pub default_width: i32,
    pub default_batch_size: i32,
    pub max_batch_size: i32,
    pub max_width: i32,
    pub vocab_size: i32,
    pub hidden_size: i32,
    pub vocab_tile_size: i32,
    pub blank_id: i32,
    pub warmup_runs: i32,
}

pub struct NativeRecognizer {
    _library: Library,
    handle: *mut c_void,
    destroy: DestroyFn,
    info_json_fn: InfoFn,
    recognize_f32_fn: RecognizeF32Fn,
    free_string: FreeStringFn,
}

unsafe impl Send for NativeRecognizer {}

#[derive(Debug, Serialize)]
pub struct FullPageNativeConfig<'a> {
    pub detector_engine_path: &'a str,
    pub detector_default_batch: i32,
    pub detector_default_height: i32,
    pub detector_default_width: i32,
    pub detector_max_batch: i32,
    pub detector_max_height: i32,
    pub detector_max_width: i32,
    pub detector_warmup_runs: i32,
    pub detector_limit_side_len: i32,
    pub detector_limit_type: &'a str,
    pub detector_max_side_limit: i32,
    pub recognizer_engine_path: &'a str,
    pub weight_path: &'a str,
    pub bias_path: &'a str,
    pub characters_path: &'a str,
    pub recognizer_default_width: i32,
    pub recognizer_default_batch_size: i32,
    pub recognizer_max_batch_size: i32,
    pub recognizer_max_width: i32,
    pub vocab_size: i32,
    pub hidden_size: i32,
    pub vocab_tile_size: i32,
    pub blank_id: i32,
    pub recognizer_warmup_runs: i32,
    pub det_thresh: f32,
    pub det_box_thresh: f32,
    pub det_unclip_ratio: f32,
    pub det_max_candidates: i32,
    pub det_min_size: i32,
    pub recognition_height: i32,
    pub recognition_buckets_csv: &'a str,
}

pub struct NativeFullPage {
    _library: Library,
    handle: *mut c_void,
    destroy: FullPageDestroyFn,
    info_json_fn: FullPageInfoFn,
    recognize_image_fn: FullPageRecognizeImageFn,
    free_string: FreeStringFn,
}

unsafe impl Send for NativeFullPage {}

impl NativeRecognizer {
    pub fn load(library_path: &Path, config: &NativeConfig<'_>) -> Result<Self> {
        let config_json = serde_json::to_string(config)?;
        let config_c = CString::new(config_json).context("native config contains NUL byte")?;

        let library = unsafe { Library::new(library_path) }
            .with_context(|| format!("failed to load native library {}", library_path.display()))?;
        let create: CreateFn = unsafe { *library.get(b"ppocrv6_recognizer_create\0")? };
        let destroy: DestroyFn = unsafe { *library.get(b"ppocrv6_recognizer_destroy\0")? };
        let info_json_fn: InfoFn = unsafe { *library.get(b"ppocrv6_recognizer_info_json\0")? };
        let recognize_f32_fn: RecognizeF32Fn =
            unsafe { *library.get(b"ppocrv6_recognizer_recognize_f32\0")? };
        let free_string: FreeStringFn =
            unsafe { *library.get(b"ppocrv6_recognizer_free_string\0")? };

        let mut handle = ptr::null_mut();
        let mut error = ptr::null_mut();
        let code = unsafe { create(config_c.as_ptr(), &mut handle, &mut error) };
        if code != 0 || handle.is_null() {
            let message = unsafe { take_native_string(error, free_string) }
                .unwrap_or_else(|| "native recognizer create failed".to_string());
            return Err(anyhow!(message));
        }

        Ok(Self {
            _library: library,
            handle,
            destroy,
            info_json_fn,
            recognize_f32_fn,
            free_string,
        })
    }

    pub fn info_json(&mut self) -> Result<Value> {
        let mut out = ptr::null_mut();
        let mut error = ptr::null_mut();
        let code = unsafe { (self.info_json_fn)(self.handle, &mut out, &mut error) };
        if code != 0 {
            let message = unsafe { take_native_string(error, self.free_string) }
                .unwrap_or_else(|| "native info call failed".to_string());
            return Err(anyhow!(message));
        }
        let raw = unsafe { take_native_string(out, self.free_string) }
            .ok_or_else(|| anyhow!("native info returned null JSON"))?;
        serde_json::from_str(&raw).context("native info was not valid JSON")
    }

    pub fn recognize_f32_json(
        &mut self,
        nchw: &[f32],
        count: i32,
        width: i32,
        batch_size: i32,
        return_timesteps: bool,
    ) -> Result<Value> {
        let mut out = ptr::null_mut();
        let mut error = ptr::null_mut();
        let code = unsafe {
            (self.recognize_f32_fn)(
                self.handle,
                nchw.as_ptr(),
                count,
                width,
                batch_size,
                i32::from(return_timesteps),
                &mut out,
                &mut error,
            )
        };
        if code != 0 {
            let message = unsafe { take_native_string(error, self.free_string) }
                .unwrap_or_else(|| "native recognize call failed".to_string());
            return Err(anyhow!(message));
        }
        let raw = unsafe { take_native_string(out, self.free_string) }
            .ok_or_else(|| anyhow!("native recognize returned null JSON"))?;
        serde_json::from_str(&raw).context("native recognize result was not valid JSON")
    }
}

impl Drop for NativeRecognizer {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            unsafe { (self.destroy)(self.handle) };
            self.handle = ptr::null_mut();
        }
    }
}

impl NativeFullPage {
    pub fn load(library_path: &Path, config: &FullPageNativeConfig<'_>) -> Result<Self> {
        let config_json = serde_json::to_string(config)?;
        let config_c =
            CString::new(config_json).context("native full-page config contains NUL byte")?;

        let library = unsafe { Library::new(library_path) }
            .with_context(|| format!("failed to load native library {}", library_path.display()))?;
        let create: FullPageCreateFn = unsafe { *library.get(b"ppocrv6_full_page_create\0")? };
        let destroy: FullPageDestroyFn = unsafe { *library.get(b"ppocrv6_full_page_destroy\0")? };
        let info_json_fn: FullPageInfoFn =
            unsafe { *library.get(b"ppocrv6_full_page_info_json\0")? };
        let recognize_image_fn: FullPageRecognizeImageFn =
            unsafe { *library.get(b"ppocrv6_full_page_recognize_image\0")? };
        let free_string: FreeStringFn =
            unsafe { *library.get(b"ppocrv6_full_page_free_string\0")? };

        let mut handle = ptr::null_mut();
        let mut error = ptr::null_mut();
        let code = unsafe { create(config_c.as_ptr(), &mut handle, &mut error) };
        if code != 0 || handle.is_null() {
            let message = unsafe { take_native_string(error, free_string) }
                .unwrap_or_else(|| "native full-page create failed".to_string());
            return Err(anyhow!(message));
        }

        Ok(Self {
            _library: library,
            handle,
            destroy,
            info_json_fn,
            recognize_image_fn,
            free_string,
        })
    }

    pub fn info_json(&mut self) -> Result<Value> {
        let mut out = ptr::null_mut();
        let mut error = ptr::null_mut();
        let code = unsafe { (self.info_json_fn)(self.handle, &mut out, &mut error) };
        if code != 0 {
            let message = unsafe { take_native_string(error, self.free_string) }
                .unwrap_or_else(|| "native full-page info call failed".to_string());
            return Err(anyhow!(message));
        }
        let raw = unsafe { take_native_string(out, self.free_string) }
            .ok_or_else(|| anyhow!("native full-page info returned null JSON"))?;
        serde_json::from_str(&raw).context("native full-page info was not valid JSON")
    }

    pub fn recognize_image_json(
        &mut self,
        image_rgb: &[u8],
        image_height: i32,
        image_width: i32,
        image_stride: i32,
    ) -> Result<Value> {
        let mut out = ptr::null_mut();
        let mut error = ptr::null_mut();
        let source_color_order_rgb = 0;
        let code = unsafe {
            (self.recognize_image_fn)(
                self.handle,
                image_rgb.as_ptr(),
                image_height,
                image_width,
                image_stride,
                source_color_order_rgb,
                &mut out,
                &mut error,
            )
        };
        if code != 0 {
            let message = unsafe { take_native_string(error, self.free_string) }
                .unwrap_or_else(|| "native full-page recognize call failed".to_string());
            return Err(anyhow!(message));
        }
        let raw = unsafe { take_native_string(out, self.free_string) }
            .ok_or_else(|| anyhow!("native full-page recognize returned null JSON"))?;
        serde_json::from_str(&raw).context("native full-page recognize result was not valid JSON")
    }
}

impl Drop for NativeFullPage {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            unsafe { (self.destroy)(self.handle) };
            self.handle = ptr::null_mut();
        }
    }
}

unsafe fn take_native_string(ptr: *mut c_char, free_string: FreeStringFn) -> Option<String> {
    if ptr.is_null() {
        return None;
    }
    let value = CStr::from_ptr(ptr).to_string_lossy().into_owned();
    free_string(ptr);
    Some(value)
}
