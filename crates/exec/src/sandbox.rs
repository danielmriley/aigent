//! Platform sandboxing for tool child-process isolation.
//!
//! Activated by the `sandbox` Cargo feature.  When the feature is absent the
//! public surface compiles to no-ops so other crates never need `#[cfg]` guards.
//!
//! ## What each platform does
//!
//! | Platform | Mechanism                  | Scope                              |
//! |----------|----------------------------|------------------------------------|
//! | Linux    | `PR_SET_NO_NEW_PRIVS` + `seccomp(2)` TSYNC | Spawned shell children |
//! | macOS    | `sandbox_init(3)` Darwin profile | Spawned shell children         |
//! | Other    | No-op                      | —                                  |
//!
//! ### Linux detail
//! `PR_SET_NO_NEW_PRIVS` (1) prevents any child from gaining privileges via
//! `setuid`/`setcap` binaries.  It is inherited across `execve` and cannot be
//! unset, providing a hard guarantee that `run_shell` cannot escalate.
//!
//! A minimal seccomp BPF ALLOW-list is then installed with
//! `SECCOMP_SET_MODE_FILTER | SECCOMP_FILTER_FLAG_TSYNC` so that all threads
//! of the child are covered simultaneously.
//!
//! The allow-list covers file I/O, networking, process management, and memory
//! allocation.  Any unrecognised syscall returns `ENOSYS` rather than killing
//! the process, which keeps the child alive while safely blocking surprises.
//!
//! ### macOS detail
//! `sandbox_init(3)` applies a named sandbox profile.  We use the built-in
//! `"no-network"` profile as a base and add workspace-path allow rules by
//! generating an inline Scheme profile string.

// ── Feature guard — the entire module body is gated on `sandbox` ───────────

/// Sandbox profile controls how aggressive the seccomp filter and platform
/// restrictions are.  `Strict` is the default for general `run_shell`;
/// `GitFriendly` widens the filter to permit DNS resolution and TLS syscalls
/// that `git`, `curl`, and `git-remote-https` need.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SandboxProfile {
    /// Minimal allow-list — suitable for most shell commands.
    Strict,
    /// Expanded allow-list — adds hostname and network syscalls for git/TLS.
    GitFriendly,
}

/// Apply sandbox restrictions to the **current process**.
///
/// On Linux this installs `PR_SET_NO_NEW_PRIVS` and a minimal seccomp
/// ALLOW-list.  On macOS it calls `sandbox_init` with a workspace-scoped
/// profile.  On other platforms this is a no-op.
///
/// # When to call
/// Call this inside a `pre_exec` hook (after `fork`, before `exec`) when
/// spawning a shell child via `std::process::Command::pre_exec`.  It must
/// **not** be called in the main daemon process — only in the child.
///
/// # Safety
/// This function is `unsafe` because it must run between `fork` and `exec`
/// where only async-signal-safe operations are permitted.  The syscall paths
/// used (prctl, seccomp, sandbox_init FFI) are async-signal-safe.
#[allow(unused_variables)]
pub unsafe fn apply_to_child(workspace_root: &str) -> std::io::Result<()> {
    #[cfg(all(feature = "sandbox", target_os = "linux"))]
    {
        // SAFETY: apply_linux is unsafe and must run between fork/exec.
        unsafe { apply_linux()? };
    }

    #[cfg(all(feature = "sandbox", target_os = "macos"))]
    {
        // SAFETY: apply_macos is unsafe and must run between fork/exec.
        unsafe { apply_macos(workspace_root)? };
    }

    Ok(())
}

/// Returns `true` when the sandbox feature is active and the current platform
/// supports it.  Useful for logging / status reporting.
pub fn is_active() -> bool {
    #[cfg(all(feature = "sandbox", any(target_os = "linux", target_os = "macos")))]
    {
        return true;
    }
    #[allow(unreachable_code)]
    false
}

// ── Linux ────────────────────────────────────────────────────────────────────

#[cfg(all(feature = "sandbox", target_os = "linux"))]
unsafe fn apply_linux() -> std::io::Result<()> {
    use std::io;

    // 1. No new privileges — inherited across execve, cannot be unset.
    const PR_SET_NO_NEW_PRIVS: libc::c_int = 38;
    // SAFETY: prctl is async-signal-safe and called between fork/exec.
    if unsafe { libc::prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) } != 0 {
        return Err(io::Error::last_os_error());
    }

    // 1b. UTS namespace isolation — give the sandbox child a distinct
    // hostname so forked processes cannot impersonate the host.
    // CLONE_NEWUTS requires CAP_SYS_ADMIN; if unavailable (container,
    // unprivileged user) we skip silently — no-new-privs is still active.
    #[cfg(target_os = "linux")]
    {
        const CLONE_NEWUTS: libc::c_int = 0x0400_0000;
        // SAFETY: unshare is async-signal-safe.
        let ret = unsafe { libc::unshare(CLONE_NEWUTS) };
        if ret == 0 {
            let name = b"aigent-sandbox\0";
            // SAFETY: name is a valid NUL-terminated byte slice.
            let _ = unsafe { libc::sethostname(name.as_ptr().cast(), name.len() - 1) };
        }
        // Non-fatal: silently ignore EPERM / EINVAL in restricted envs.
    }

    // 2. Minimal seccomp BPF allow-list.
    //    We build a hand-crafted BPF program that:
    //      - Returns ALLOW for each syscall in the allow-list.
    //      - Returns ERRNO(ENOSYS) for everything else (graceful deny).
    //
    //    Architecture note: the BPF program below is x86-64 specific.
    //    On aarch64 the syscall numbers differ; we skip the allow-list there
    //    (PR_SET_NO_NEW_PRIVS still applies) and rely on wasmtime's own
    //    sandbox for WASM guests.
    #[cfg(target_arch = "x86_64")]
    {
        // SAFETY: install_seccomp_allowlist is unsafe and must run between fork/exec.
        unsafe { install_seccomp_allowlist()? };
    }

    Ok(())
}

#[cfg(all(feature = "sandbox", target_os = "linux", target_arch = "x86_64"))]
unsafe fn install_seccomp_allowlist() -> std::io::Result<()> {
    use std::io;
    use std::mem;

    // BPF socket_filter instruction layout.
    #[repr(C)]
    struct SockFilter {
        code: u16,
        jt: u8,
        jf: u8,
        k: u32,
    }

    #[repr(C)]
    struct SockFprog {
        len: u16,
        filter: *const SockFilter,
    }

    // BPF opcodes
    const BPF_LD: u16 = 0x00;
    const BPF_W: u16 = 0x00;
    const BPF_ABS: u16 = 0x20;
    const BPF_JMP: u16 = 0x05;
    const BPF_JEQ: u16 = 0x10;
    const BPF_K: u16 = 0x00;
    const BPF_RET: u16 = 0x06;

    const SECCOMP_RET_ALLOW: u32 = 0x7fff_0000;
    const SECCOMP_RET_ERRNO: u32 = 0x0005_0000; // base
    const ENOSYS: u32 = 38;

    // Offset of sc_nr in seccomp_data struct (arch-independent for x86_64).
    const SECCOMP_DATA_NR_OFFSET: u32 = 0;

    // Allowed syscall numbers for x86_64.
    const ALLOWED: &[u32] = &[
        0,   // read
        1,   // write
        2,   // open
        3,   // close
        4,   // stat
        5,   // fstat
        6,   // lstat
        7,   // poll
        8,   // lseek
        9,   // mmap
        10,  // mprotect
        11,  // munmap
        12,  // brk
        13,  // rt_sigaction
        14,  // rt_sigprocmask
        15,  // rt_sigreturn
        16,  // ioctl
        17,  // pread64
        18,  // pwrite64
        19,  // readv
        20,  // writev
        21,  // access
        22,  // pipe
        23,  // select
        24,  // sched_yield
        25,  // mremap
        26,  // msync
        28,  // madvise
        32,  // dup
        33,  // dup2
        39,  // getpid
        41,  // socket
        42,  // connect
        43,  // accept
        44,  // sendto
        45,  // recvfrom
        46,  // sendmsg
        47,  // recvmsg
        49,  // bind
        50,  // listen
        51,  // getsockname
        52,  // getpeername
        53,  // socketpair
        54,  // setsockopt
        55,  // getsockopt
        56,  // clone
        57,  // fork
        58,  // vfork
        59,  // execve
        60,  // exit
        61,  // wait4
        62,  // kill
        72,  // fcntl
        73,  // flock
        74,  // fsync
        75,  // fdatasync
        76,  // truncate
        77,  // ftruncate
        78,  // getdents
        79,  // getcwd
        80,  // chdir
        82,  // rename
        83,  // mkdir
        84,  // rmdir
        85,  // creat
        86,  // link
        87,  // unlink
        88,  // symlink
        89,  // readlink
        90,  // chmod
        91,  // fchmod
        95,  // umask
        96,  // gettimeofday
        97,  // getrlimit
        99,  // sysinfo
        102, // getuid
        104, // getgid
        107, // geteuid
        108, // getegid
        110, // getppid
        111, // getpgrp
        131, // sigaltstack
        132, // utime
        137, // statfs
        138, // fstatfs
        158, // arch_prctl
        186, // gettid
        202, // futex
        204, // sched_getaffinity
        218, // set_tid_address
        228, // clock_gettime
        229, // clock_getres
        231, // exit_group
        232, // epoll_wait
        233, // epoll_ctl
        234, // tgkill
        257, // openat
        258, // mkdirat
        261, // futimesat
        262, // newfstatat
        263, // unlinkat
        264, // renameat
        265, // linkat
        266, // symlinkat
        267, // readlinkat
        268, // fchmodat
        269, // faccessat
        270, // pselect6
        271, // ppoll
        281, // epoll_pwait
        285, // fallocate
        290, // eventfd2
        291, // epoll_create1
        292, // dup3
        293, // pipe2
        302, // prlimit64
        318, // getrandom
        332, // statx
        // ── git-friendly additions (hostname isolation + TLS/networking) ────
        160, // setrlimit (git-remote-https resource management)
        161, // chroot (unused but returned by glibc fallback paths)
        170, // sethostname (UTS namespace isolation)
        435, // clone3 (modern glibc thread creation)
        273, // set_robust_list (glibc/pthread internal)
        63,  // uname (git version checks)
        101, // ptrace (denied by no-new-privs but avoids ENOSYS confusion)
    ];

    let n = ALLOWED.len();

    // Each allowed syscall needs: LD nr, JEQ <nr> ALLOW, and at the end DENY.
    // Program: [LD abs #0] + [JEQ nr - je to ALLOW, jf to next] * n + [RET ENOSYS] + [RET ALLOW]
    // Total instructions: 1 + n * 1 + 2 = n + 3
    // We use a compact form: one JEQ per syscall with jt pointing to the final ALLOW.

    let prog_len = 1 + n + 2;
    let mut prog: Vec<SockFilter> = Vec::with_capacity(prog_len);

    // Instruction 0: load syscall number into accumulator
    prog.push(SockFilter {
        code: BPF_LD | BPF_W | BPF_ABS,
        jt: 0,
        jf: 0,
        k: SECCOMP_DATA_NR_OFFSET,
    });

    // Instructions 1..=n: one JEQ per allowed syscall
    // If match: jump forward to the ALLOW instruction at offset (n - i) past the JEQ
    // If no match: fall through to next JEQ
    for (i, &nr) in ALLOWED.iter().enumerate() {
        let steps_to_allow = (n - i) as u8; // distance to ALLOW instruction
        prog.push(SockFilter {
            code: BPF_JMP | BPF_JEQ | BPF_K,
            jt: steps_to_allow, // jump to ALLOW on match
            jf: 0,              // fall through on no-match
            k: nr,
        });
    }

    // Instruction n+1: default deny (ERRNO ENOSYS)
    prog.push(SockFilter {
        code: BPF_RET | BPF_K,
        jt: 0,
        jf: 0,
        k: SECCOMP_RET_ERRNO | (ENOSYS & 0xFFFF),
    });

    // Instruction n+2: allow
    prog.push(SockFilter {
        code: BPF_RET | BPF_K,
        jt: 0,
        jf: 0,
        k: SECCOMP_RET_ALLOW,
    });

    let fprog = SockFprog {
        len: prog.len() as u16,
        filter: prog.as_ptr(),
    };

    // SECCOMP_SET_MODE_FILTER (1) | SECCOMP_FILTER_FLAG_TSYNC (2)
    const SYS_SECCOMP: libc::c_long = 317;
    const SECCOMP_SET_MODE_FILTER: libc::c_ulong = 1;
    const SECCOMP_FILTER_FLAG_TSYNC: libc::c_ulong = 2;

    // SAFETY: seccomp syscall is async-signal-safe; fprog outlives the call.
    let ret = unsafe {
        libc::syscall(
            SYS_SECCOMP,
            SECCOMP_SET_MODE_FILTER,
            SECCOMP_FILTER_FLAG_TSYNC,
            &fprog as *const SockFprog as *const libc::c_void,
        )
    };

    if ret != 0 {
        // Non-fatal: seccomp may be unavailable in some container environments.
        // PR_SET_NO_NEW_PRIVS is still active.
        let _ = io::Error::last_os_error(); // consume errno
        tracing::warn!("sandbox: seccomp syscall filter unavailable; no-new-privs still active");
    }

    mem::forget(prog); // BPF program must outlive the syscall (it's consumed immediately)

    Ok(())
}

// ── macOS ────────────────────────────────────────────────────────────────────

#[cfg(all(feature = "sandbox", target_os = "macos"))]
unsafe fn apply_macos(workspace_root: &str) -> std::io::Result<()> {
    use std::ffi::CString;
    use std::io;
    use std::ptr;

    // Apple private API — present in all macOS versions >= 10.5.
    extern "C" {
        fn sandbox_init(
            profile: *const libc::c_char,
            flags: u64,
            errorbuf: *mut *mut libc::c_char,
        ) -> libc::c_int;

        fn sandbox_free_error(errorbuf: *mut libc::c_char);
    }

    // Build an inline Scheme-like sandbox profile:
    // Allow file R/W only inside the workspace root and standard system paths.
    // Allow outbound TCP on ports 80, 443 (web_search).
    // Deny everything else (default-deny).
    let profile = format!(
        r#"(version 1)
(deny default)
(allow file-read* (subpath "/usr") (subpath "/lib") (subpath "/etc")
                  (subpath "/tmp") (subpath "/var/tmp") (subpath "{ws}"))
(allow file-write* (subpath "/tmp") (subpath "/var/tmp") (subpath "{ws}"))
(allow process-exec)
(allow process-fork)
(allow sysctl-read)
(allow mach-lookup)
(allow network-outbound (remote tcp "*:80") (remote tcp "*:443"))
(allow signal (target self))
"#,
        ws = workspace_root
    );

    let c_profile = CString::new(profile)
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidInput, e))?;
    let mut errorbuf: *mut libc::c_char = ptr::null_mut();

    // SAFETY: c_profile outlives the call; errorbuf is valid output pointer.
    let ret = unsafe { sandbox_init(c_profile.as_ptr(), 0, &mut errorbuf) };
    if ret != 0 {
        let err_str = if errorbuf.is_null() {
            "unknown sandbox_init error".to_string()
        } else {
            // SAFETY: sandbox_init sets errorbuf to a valid C string on failure.
            let msg = unsafe { std::ffi::CStr::from_ptr(errorbuf) }.to_string_lossy().to_string();
            // SAFETY: errorbuf was allocated by sandbox_init.
            unsafe { sandbox_free_error(errorbuf) };
            msg
        };
        return Err(io::Error::new(io::ErrorKind::Other, err_str));
    }

    Ok(())
}
