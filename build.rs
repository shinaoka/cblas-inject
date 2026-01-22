fn main() {
    #[cfg(feature = "openblas")]
    {
        // macOS homebrew path
        if cfg!(target_os = "macos") {
            if cfg!(target_arch = "aarch64") {
                println!("cargo:rustc-link-search=/opt/homebrew/opt/openblas/lib");
            } else {
                println!("cargo:rustc-link-search=/usr/local/opt/openblas/lib");
            }
        }
    }
}
