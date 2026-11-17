use std::env;
use std::fs;
use std::path::{Path, PathBuf};

fn main() {
    println!("cargo:rerun-if-env-changed=LLM_COMPRESS_LLAMA_LIB_DIR");
    let target = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    let target_arch = env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();
    let configured = env::var_os("LLM_COMPRESS_LLAMA_LIB_DIR").map(PathBuf::from);
    let candidates = configured
        .into_iter()
        .chain(default_dirs(&target, &target_arch));

    for directory in candidates {
        if let Some(library) = find_library(&directory, &target) {
            let link_dir =
                PathBuf::from(env::var_os("OUT_DIR").expect("OUT_DIR is set")).join("llama-link");
            fs::create_dir_all(&link_dir).expect("create llama link directory");
            prepare_link_name(&library, &link_dir, &target);
            if target == "linux" {
                let ggml = configured_ggml_dir(&directory, &target_arch)
                    .and_then(|dir| find_named_library(&dir, "libggml.so"))
                    .or_else(|| {
                        default_dirs(&target, &target_arch)
                            .iter()
                            .find_map(|dir| find_named_library(dir, "libggml.so"))
                    })
                    .expect("libggml.so is required by llama.cpp");
                let link = link_dir.join("libggml.so");
                let _ = fs::remove_file(&link);
                #[cfg(unix)]
                std::os::unix::fs::symlink(&ggml, &link)
                    .or_else(|_| fs::copy(&ggml, &link).map(|_| ()))
                    .expect("make ggml linker name");
                println!("cargo:rustc-link-lib=ggml");
                if let Some(ggml_dir) = ggml.parent() {
                    println!("cargo:rustc-link-arg=-Wl,-rpath,{}", ggml_dir.display());
                }
                println!("cargo:rerun-if-changed={}", ggml.display());
            }
            println!("cargo:rustc-link-search=native={}", link_dir.display());
            println!("cargo:rustc-link-arg=-Wl,-rpath,{}", directory.display());
            println!("cargo:rerun-if-changed={}", library.display());
            return;
        }
    }

    panic!(
        "llama.cpp shared library not found. Install the llama.cpp development library or set LLM_COMPRESS_LLAMA_LIB_DIR."
    );
}

fn configured_ggml_dir(llama_dir: &Path, arch: &str) -> Option<PathBuf> {
    if find_named_library(llama_dir, "libggml.so").is_some() {
        return Some(llama_dir.to_path_buf());
    }
    let parent = llama_dir.parent()?;
    [
        parent.to_path_buf(),
        parent.join(format!("{arch}-linux-gnu")),
    ]
    .into_iter()
    .find(|dir| find_named_library(dir, "libggml.so").is_some())
}

fn find_named_library(directory: &Path, base_name: &str) -> Option<PathBuf> {
    let exact = directory.join(base_name);
    if exact.is_file() {
        return Some(exact);
    }
    let prefix = format!("{base_name}.");
    fs::read_dir(directory)
        .ok()?
        .flatten()
        .map(|entry| entry.path())
        .find(|path| {
            path.file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| name.starts_with(&prefix))
        })
}

fn default_dirs(os: &str, arch: &str) -> Vec<PathBuf> {
    match os {
        "linux" => vec![
            PathBuf::from(format!("/usr/lib/{arch}-linux-gnu/llama")),
            PathBuf::from(format!("/usr/lib/{arch}-linux-gnu")),
            PathBuf::from("/usr/local/lib"),
            PathBuf::from("/usr/lib"),
        ],
        "macos" => vec![
            PathBuf::from("/opt/homebrew/lib"),
            PathBuf::from("/usr/local/lib"),
        ],
        "windows" => vec![
            PathBuf::from("C:/llama.cpp/bin"),
            PathBuf::from("C:/llama.cpp/lib"),
        ],
        _ => Vec::new(),
    }
}

fn find_library(directory: &Path, os: &str) -> Option<PathBuf> {
    let names: &[&str] = match os {
        "linux" => &["libllama.so", "libllama.so.0"],
        "macos" => &["libllama.dylib"],
        "windows" => &["llama.dll", "llama.lib", "libllama.dll.a"],
        _ => return None,
    };
    names
        .iter()
        .map(|name| directory.join(name))
        .find(|path| path.is_file())
}

fn prepare_link_name(library: &Path, link_dir: &Path, os: &str) {
    if os == "windows" {
        return;
    }
    let link_name = match os {
        "linux" => "libllama.so",
        "macos" => "libllama.dylib",
        _ => return,
    };
    let link = link_dir.join(link_name);
    let _ = fs::remove_file(&link);
    #[cfg(unix)]
    std::os::unix::fs::symlink(library, &link)
        .or_else(|_| fs::copy(library, &link).map(|_| ()))
        .expect("make llama linker name");
}
