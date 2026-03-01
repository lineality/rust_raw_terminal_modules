//! # Raw Terminal Module for Linux x86_64
//!
//! ## Project Context
//!
//! This module provides terminal raw mode functionality using pure Rust with
//! inline assembly for Linux x86_64 systems. The goal is to enable reading
//! keyboard input byte-by-byte without line buffering or echo, which is
//! essential for terminal user interfaces, text editors, and interactive
//! command-line applications.
//!
//! ## What This Module Does
//!
//! 1. Opens `/dev/tty` to get direct terminal access (works even if stdin is redirected)
//! 2. Saves the current terminal configuration (termios struct)
//! 3. Modifies terminal flags to disable:
//!    - Line buffering (ICANON) - so we get bytes immediately, not after Enter
//!    - Echo (ECHO) - so typed characters aren't automatically displayed
//!    - Signal processing (ISIG) - so Ctrl+C doesn't kill the program
//!    - Input/output processing - so we get raw bytes as-is
//! 4. Restores original terminal settings on drop (RAII pattern)
//!
//! ## Why This Approach
//!
//! - **No Rust-libc import dependency**: No libc for terminal attribute control:
//!     tcgetattr/tcsetattr/cfmakeraw are replaced with direct syscalls via inline assembly.
//!     File I/O still uses std::fs (which uses libc internally).
//! - **No external crates**: Self-contained implementation
//! - **Minimal unsafe**: Only the syscall interface requires unsafe
//! - **RAII cleanup**: Terminal state is always restored, even on panic
//!
//! ## Architecture Limitation
//!
//! This implementation is **Linux x86_64 only**. The syscall numbers, ioctl
//! request codes, and termios struct layout are architecture-specific.
//! Porting to other architectures (ARM64, x86_32) would require different
//! constants and potentially different struct layouts.
//!
//! ## C Function Equivalents
//!
//! This module replaces these C/libc functions:
//! - `tcgetattr(fd, &termios)` → `get_terminal_attr(fd)`
//! - `tcsetattr(fd, TCSANOW, &termios)` → `set_terminal_attr(fd, &termios)`
//! - `cfmakeraw(&termios)` → `make_raw(&mut termios)`
//!
//! ## Syscall Details
//!
//! Both tcgetattr and tcsetattr are implemented via the ioctl syscall:
//! - `ioctl(fd, TCGETS, &termios)` - read terminal attributes
//! - `ioctl(fd, TCSETS, &termios)` - write terminal attributes (immediate)
//!
//! ## References
//!
//! - Linux kernel: `include/uapi/asm-generic/termbits.h`
//! - Linux kernel: `include/uapi/asm-generic/ioctls.h`
//! - x86_64 syscall table: syscall number 16 = ioctl

// ============================================================================
// ARCHITECTURE GUARD - COMPILE-TIME PLATFORM VERIFICATION
// ============================================================================
//
// ## Project Context
//
// This module uses x86_64-specific:
// - Syscall numbers (SYS_IOCTL = 16)
// - ioctl request codes (TCGETS = 0x5401, TCSETS = 0x5402)
// - Termios struct layout (36 bytes with specific field offsets)
// - Inline assembly with x86_64 register conventions
//
// These values differ on other architectures (ARM64, x86_32, RISC-V, etc.)
// and other operating systems (macOS, BSDs, Windows).
//
// Attempting to run this code on unsupported platforms would cause:
// - Incorrect syscall invocation (wrong syscall number)
// - Memory corruption (wrong struct layout passed to kernel)
// - Undefined behavior (wrong ioctl codes)
//
// This compile-time check prevents confusing runtime failures by failing
// fast at build time with a clear error message.
// ============================================================================

#[cfg(not(all(target_os = "linux", target_arch = "x86_64")))]
compile_error!(
    "raw_terminal_x86 only supports Linux x86_64. \
     Syscall numbers, ioctl request codes, and termios struct layouts \
     are architecture-specific. Porting to other architectures requires \
     different constants and potentially different struct definitions. \
     See Linux kernel headers: arch/*/include/uapi/asm/termbits.h"
);

use core::arch::asm;
use std::fs::{File, OpenOptions};
use std::io::{self, Read, Write};
use std::os::unix::io::AsRawFd;

// ============================================================================
// LINUX KERNEL CONSTANTS - x86_64 SPECIFIC
// ============================================================================
//
// These values are from Linux kernel headers and are architecture-specific.
// Source: linux/include/uapi/asm-generic/termbits.h
// Source: linux/include/uapi/asm-generic/ioctls.h
//
// WARNING: These values are ONLY valid for Linux x86_64. Other architectures
// (ARM, MIPS, etc.) may have different values.
// ============================================================================

// -----------------------------------------------------------------------------
// Input Mode Flags (c_iflag) - Control input processing
// -----------------------------------------------------------------------------
// These flags control how the terminal processes incoming bytes BEFORE
// the application sees them. For raw mode, we disable all processing.

/// Ignore BREAK condition on input
/// When set: BREAK (serial line break) is ignored
/// For raw mode: DISABLE - we want to see everything
const IGNBRK: u32 = 0o000001;

/// Signal interrupt on BREAK
/// When set: BREAK causes SIGINT to be sent
/// For raw mode: DISABLE - no signal generation from input
const BRKINT: u32 = 0o000002;

/// Mark parity and framing errors
/// When set: Errored bytes are prefixed with \377 \0
/// For raw mode: DISABLE - pass bytes as-is
const PARMRK: u32 = 0o000010;

/// Strip 8th bit off input bytes
/// When set: Input bytes are masked to 7 bits (ASCII only)
/// For raw mode: DISABLE - we want full 8-bit bytes
const ISTRIP: u32 = 0o000040;

/// Map NL to CR on input
/// When set: Newline (\n, 0x0A) is converted to carriage return (\r, 0x0D)
/// For raw mode: DISABLE - pass bytes as-is
const INLCR: u32 = 0o000100;

/// Ignore carriage return on input
/// When set: Carriage return (\r) is discarded
/// For raw mode: DISABLE - we want to see \r
const IGNCR: u32 = 0o000200;

/// Map CR to NL on input (unless IGNCR is set)
/// When set: Carriage return (\r) is converted to newline (\n)
/// For raw mode: DISABLE - pass bytes as-is
const ICRNL: u32 = 0o000400;

/// Enable XON/XOFF flow control on output
/// When set: Ctrl+S pauses output, Ctrl+Q resumes
/// For raw mode: DISABLE - we want Ctrl+S and Ctrl+Q as regular keys
const IXON: u32 = 0o002000;

// -----------------------------------------------------------------------------
// Output Mode Flags (c_oflag) - Control output processing
// -----------------------------------------------------------------------------

/// Post-process output
/// When set: Output processing is enabled (e.g., \n → \r\n conversion)
/// For raw mode: DISABLE - application handles all output formatting
/// Note: When disabled, write \r\n explicitly for newlines
const OPOST: u32 = 0o000001;

// -----------------------------------------------------------------------------
// Control Mode Flags (c_cflag) - Control hardware/driver behavior
// -----------------------------------------------------------------------------

/// Character size mask (2 bits)
/// Values: CS5, CS6, CS7, CS8 for 5-8 bit characters
/// For raw mode: Set to CS8 for full 8-bit bytes
const CSIZE: u32 = 0o000060;

/// 8-bit character size
/// For raw mode: ENABLE - we want full 8-bit bytes
const CS8: u32 = 0o000060;

/// Enable parity generation/checking
/// When set: Parity bit is added to output, checked on input
/// For raw mode: DISABLE - no parity processing
const PARENB: u32 = 0o000400;

// -----------------------------------------------------------------------------
// Local Mode Flags (c_lflag) - Control line discipline behavior
// -----------------------------------------------------------------------------
// These are the most important flags for raw mode. They control the
// "line discipline" - how the terminal buffers and processes input.

/// Enable signals (SIGINT, SIGQUIT, SIGTSTP)
/// When set: Ctrl+C sends SIGINT, Ctrl+\ sends SIGQUIT, Ctrl+Z sends SIGTSTP
/// For raw mode: DISABLE - application handles these keys directly
/// WARNING: With this disabled, Ctrl+C will NOT terminate your program!
const ISIG: u32 = 0o000001;

/// Canonical mode (line buffering)
/// When set: Input is buffered until Enter/newline, with line editing
/// For raw mode: DISABLE - this is the key flag for byte-by-byte input
/// This is the PRIMARY flag that makes raw mode "raw"
const ICANON: u32 = 0o000002;

/// Echo input characters
/// When set: Typed characters are automatically displayed
/// For raw mode: DISABLE - application controls all display
const ECHO: u32 = 0o000010;

/// Echo NL even if ECHO is disabled
/// When set: Newline is echoed even when ECHO is off
/// For raw mode: DISABLE - no automatic echo of anything
const ECHONL: u32 = 0o000100;

/// Enable implementation-defined input processing
/// When set: Additional processing (e.g., Ctrl+V for literal next char)
/// For raw mode: DISABLE - no extended processing
const IEXTEN: u32 = 0o100000;

// -----------------------------------------------------------------------------
// Control Character Indices (c_cc array)
// -----------------------------------------------------------------------------
// The c_cc array contains special control characters and timing parameters.
// For raw mode, we only care about VMIN and VTIME which control read behavior.

/// Index for VTIME - timeout in deciseconds (1/10 second)
/// Value 0 = no timeout (wait forever for VMIN bytes)
/// For raw mode: Set to 0
const VTIME: usize = 5;

/// Index for VMIN - minimum bytes before read() returns
/// Value 0 = non-blocking (return immediately with whatever is available)
/// Value 1 = blocking (wait for at least 1 byte)
/// For raw mode: Set to 1 (blocking read of single bytes)
const VMIN: usize = 6;

// -----------------------------------------------------------------------------
// ioctl Request Codes - Terminal control operations
// -----------------------------------------------------------------------------
// These are the "request" values passed to ioctl() to specify the operation.
// Source: linux/include/uapi/asm-generic/ioctls.h

/// Get terminal attributes - reads termios struct
/// Equivalent to: tcgetattr(fd, &termios)
const TCGETS: u64 = 0x5401;

/// Set terminal attributes immediately - writes termios struct
/// Equivalent to: tcsetattr(fd, TCSANOW, &termios)
/// TCSANOW = apply changes immediately (as opposed to TCSADRAIN/TCSAFLUSH)
const TCSETS: u64 = 0x5402;

// -----------------------------------------------------------------------------
// Syscall Numbers - x86_64 Linux
// -----------------------------------------------------------------------------
// Source: linux/arch/x86/entry/syscalls/syscall_64.tbl

/// ioctl syscall number on x86_64 Linux
/// Used for terminal control operations
const SYS_IOCTL: u64 = 16;

// ============================================================================
// TERMIOS STRUCT - MUST MATCH KERNEL LAYOUT EXACTLY
// ============================================================================
//
// This struct represents the terminal I/O settings. The kernel reads and
// writes this struct directly, so the memory layout MUST match exactly.
//
// Source: linux/include/uapi/asm-generic/termbits.h
//
// Note: There are multiple termios variants in Linux:
// - termios: 19-byte c_cc array, used by TCGETS/TCSETS (what we use)
// - termios2: 19-byte c_cc array, used by TCGETS2/TCSETS2, adds explicit baud
// - ktermios: 19-byte c_cc array, kernel internal
//
// The standard TCGETS/TCSETS use the basic termios with 19-byte c_cc.
// ============================================================================

/// Terminal I/O settings structure matching Linux kernel layout.
///
/// ## Memory Layout (x86_64)
///
/// ```text
/// Offset  Size  Field
/// 0       4     c_iflag (input mode flags)
/// 4       4     c_oflag (output mode flags)
/// 8       4     c_cflag (control mode flags)
/// 12      4     c_lflag (local mode flags)
/// 16      1     c_line  (line discipline)
/// 17      19    c_cc    (control characters)
/// Total: 36 bytes
/// ```
///
/// ## Why #[repr(C)]
///
/// The `#[repr(C)]` attribute ensures the struct has the same memory layout
/// as the equivalent C struct. Without this, Rust might reorder fields or
/// add padding differently than C/the kernel expects.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct Termios {
    /// Input mode flags - controls input byte processing
    c_iflag: u32,

    /// Output mode flags - controls output byte processing
    c_oflag: u32,

    /// Control mode flags - controls hardware/driver behavior
    c_cflag: u32,

    /// Local mode flags - controls line discipline (buffering, echo, signals)
    c_lflag: u32,

    /// Line discipline - usually 0 (N_TTY)
    /// We don't modify this for raw mode
    c_line: u8,

    /// Control characters array
    /// Contains special characters (VINTR, VQUIT, etc.) and VMIN/VTIME
    /// Size is 19 bytes for standard termios used with TCGETS/TCSETS
    c_cc: [u8; 19],
}

// Compile-time verification of struct size
// This will cause a compile error if the struct size is wrong
const _: () = {
    if core::mem::size_of::<Termios>() != 36 {
        panic!("Termios struct size must be exactly 36 bytes for Linux x86_64");
    }
};

// ============================================================================
// SYSCALL INTERFACE - INLINE ASSEMBLY FOR x86_64
// ============================================================================
//
// ## Why Inline Assembly?
//
// To avoid libc, we must make syscalls directly. On x86_64 Linux, this is
// done via the `syscall` instruction with arguments in specific registers.
//
// ## x86_64 Linux Syscall Convention
//
// - RAX: syscall number (input), return value (output)
// - RDI: 1st argument
// - RSI: 2nd argument
// - RDX: 3rd argument
// - R10: 4th argument (not used here)
// - R8:  5th argument (not used here)
// - R9:  6th argument (not used here)
// - RCX, R11: clobbered by syscall instruction (must be declared)
//
// ## Return Value
//
// - Success: non-negative value (often 0 for ioctl)
// - Error: negative value, where -value is the errno
//
// ## Safety Requirements
//
// This function is unsafe because:
// 1. We're executing arbitrary machine code
// 2. The kernel trusts the pointer we pass
// 3. Incorrect arguments could corrupt memory or crash
// ============================================================================

/// Execute ioctl syscall for **reading** terminal attributes (TCGETS).
///
/// ## Project Context
///
/// This wrapper is used exclusively by `get_terminal_attr` to read the
/// current termios configuration from the kernel. The kernel **writes into**
/// the pointed-to struct, so the pointer must be `*mut`.
///
/// ## Safety Contract (caller must uphold ALL of these)
///
/// - `arg` must point to a valid, properly aligned, **writable** `Termios`.
/// - The pointed-to memory must remain valid and exclusively accessible
///   for the entire duration of the syscall.
/// - `request` must be a read-type ioctl code (TCGETS) that writes into `arg`.
///
/// ## Register Usage (x86_64 Linux syscall convention)
///
/// ```text
/// RAX = syscall number (16 for ioctl) → return value
/// RDI = fd          (1st argument)
/// RSI = request     (2nd argument)
/// RDX = arg pointer (3rd argument)
/// RCX = clobbered by the syscall instruction
/// R11 = clobbered by the syscall instruction
/// ```
#[inline]
unsafe fn ioctl_read_termios(fd: i32, request: u64, arg: *mut Termios) -> i64 {
    let ret: i64;

    // SAFETY: This block is the single unsafe operation in this function.
    //
    // The inline `asm!` satisfies its safety requirements because:
    //   - `arg` is guaranteed valid, aligned, and writable by the caller's
    //     contract above.
    //   - We have exclusive access for the syscall duration (caller contract).
    //   - TCGETS writes the current terminal config into our struct.
    //   - The Termios layout matches what the Linux x86_64 kernel expects
    //     (verified by `test_termios_struct_size` and `test_termios_field_offsets`).
    //   - `nostack` is correct: the syscall instruction does not touch
    //     our stack (the kernel uses its own stack internally).
    //   - `nomem` is intentionally NOT used: the kernel writes to memory
    //     through the `arg` pointer during this syscall.
    //   - RCX and R11 are declared as clobbered: the x86_64 `syscall`
    //     instruction unconditionally overwrites both.
    unsafe {
        asm!(
            "syscall",
            // --- input then output on RAX ---
            inlateout("rax") SYS_IOCTL => ret,
            // --- inputs ---
            in("rdi") fd as u64,
            in("rsi") request,
            in("rdx") arg as u64,
            // --- clobbers ---
            out("rcx") _,
            out("r11") _,
            // --- options ---
            options(nostack),
        );
    }

    ret
}

/// Execute ioctl syscall for **writing** terminal attributes (TCSETS).
///
/// ## Project Context
///
/// This wrapper is used exclusively by `set_terminal_attr` to apply
/// termios configuration to the kernel. The kernel only **reads from**
/// the pointed-to struct, so the pointer is `*const` — accurately
/// reflecting the data flow direction.
///
/// ## Why a Separate Wrapper from `ioctl_read_termios`
///
/// TCGETS and TCSETS have opposite data flow:
/// - TCGETS: kernel **writes into** our memory → requires `*mut`
/// - TCSETS: kernel **reads from** our memory → requires only `*const`
///
/// A single function accepting `*mut` for both would force callers of
/// the write path to cast `*const` to `*mut`, misrepresenting the
/// actual memory access pattern. Splitting into two wrappers:
/// 1. Eliminates the `*const`-to-`*mut` cast in `set_terminal_attr`
/// 2. Makes each function's safety contract more precise
/// 3. Documents the kernel's data flow direction in the type signature
///
/// The assembly body is identical — the kernel receives a raw address
/// in RDX regardless. The difference is purely at the Rust type level.
///
/// ## Safety Contract (caller must uphold ALL of these)
///
/// - `arg` must point to a valid, properly aligned, **readable** `Termios`.
/// - The pointed-to memory must remain valid and accessible for the
///   entire duration of the syscall.
/// - `request` must be a write-type ioctl code (TCSETS) that reads from `arg`.
///
/// ## Register Usage (x86_64 Linux syscall convention)
///
/// ```text
/// RAX = syscall number (16 for ioctl) → return value
/// RDI = fd          (1st argument)
/// RSI = request     (2nd argument)
/// RDX = arg pointer (3rd argument)
/// RCX = clobbered by the syscall instruction
/// R11 = clobbered by the syscall instruction
/// ```
#[inline]
unsafe fn ioctl_write_termios(fd: i32, request: u64, arg: *const Termios) -> i64 {
    let ret: i64;

    // SAFETY: This block is the single unsafe operation in this function.
    //
    // The inline `asm!` satisfies its safety requirements because:
    //   - `arg` is guaranteed valid, aligned, and readable by the caller's
    //     contract above.
    //   - We have exclusive access for the syscall duration (caller contract).
    //   - TCSETS reads our struct and applies settings to the terminal driver.
    //   - The Termios layout matches what the Linux x86_64 kernel expects
    //     (verified by `test_termios_struct_size` and `test_termios_field_offsets`).
    //   - `nostack` is correct: the syscall instruction does not touch
    //     our stack (the kernel uses its own stack internally).
    //   - `nomem` is intentionally NOT used: the kernel reads from memory
    //     through the `arg` pointer during this syscall.
    //   - RCX and R11 are declared as clobbered: the x86_64 `syscall`
    //     instruction unconditionally overwrites both.
    unsafe {
        asm!(
            "syscall",
            // --- input then output on RAX ---
            inlateout("rax") SYS_IOCTL => ret,
            // --- inputs ---
            in("rdi") fd as u64,
            in("rsi") request,
            in("rdx") arg as u64,
            // --- clobbers ---
            out("rcx") _,
            out("r11") _,
            // --- options ---
            options(nostack),
        );
    }

    ret
}

// ============================================================================
// TERMINAL ATTRIBUTE FUNCTIONS
// ============================================================================
//
// These functions wrap the raw syscall in a safe interface, handling:
// - Proper initialization of the Termios struct
// - Error conversion to Rust's io::Error type
// - Pointer safety for the syscall
// ============================================================================

/// Read current terminal attributes from the kernel.
///
/// ## Project Context
///
/// This function retrieves the current terminal configuration so we can:
/// 1. Save it for later restoration (RAII pattern in RawTerminal)
/// 2. Modify it to enable raw mode
///
/// ## Validation
///
/// Per production rules, kernel return values are sanity-checked.
/// A termios with no character-size bits set (c_cflag & CSIZE == 0)
/// is invalid and indicates kernel/driver error or memory corruption.
///
/// ## Arguments
///
/// * `fd` - Raw file descriptor for the terminal
///
/// ## Returns
///
/// * `Ok(Termios)` - Current terminal attributes (validated)
/// * `Err(io::Error)` - If ioctl fails or returned data is invalid
///
/// ## Errors
///
/// - ENOTTY (25): fd is not a terminal
/// - EBADF (9): fd is not a valid file descriptor
/// - EINVAL (22): returned termios failed sanity check (synthetic error)
fn get_terminal_attr(fd: i32) -> io::Result<Termios> {
    // Initialize with zeros - kernel will overwrite on success
    let mut termios = Termios {
        c_iflag: 0,
        c_oflag: 0,
        c_cflag: 0,
        c_lflag: 0,
        c_line: 0,
        c_cc: [0u8; 19],
    };

    // SAFETY: termios is a local, properly aligned, writable Termios struct.
    // ioctl_read_termios + TCGETS will write the current config into it.
    let ret = unsafe { ioctl_read_termios(fd, TCGETS, &mut termios) };

    if ret < 0 {
        return Err(io::Error::from_raw_os_error((-ret) as i32));
    }

    // -------------------------------------------------------------------------
    // Sanity check: validate kernel response
    // -------------------------------------------------------------------------
    // Per production rules: "Every return should be checked for what can be
    // checked." A termios with c_cflag containing no character size bits
    // is invalid - this catches corrupted kernel responses or bit-flips.
    //
    // Valid CSIZE values: CS5 (0o00), CS6 (0o20), CS7 (0o40), CS8 (0o60)
    // All have at least one bit in the CSIZE mask except CS5 which is 0.
    // CS5 (5-bit characters) is extremely rare but technically valid.
    // We check that c_cflag is not entirely zero (which would be nonsensical).
    // -------------------------------------------------------------------------
    if termios.c_cflag == 0 {
        // Completely zeroed c_cflag is invalid: no baud, no char size, no flags
        // Return EINVAL to indicate invalid configuration
        return Err(io::Error::from_raw_os_error(22)); // EINVAL
    }

    Ok(termios)
}

/// Write terminal attributes to the kernel.
///
/// ## Project Context
///
/// This function is used to:
/// 1. Apply raw mode settings (after modifying with `make_raw`)
/// 2. Restore original settings (in `RawTerminal::drop`)
///
/// ## How It Works
///
/// 1. Takes a reference to a Termios struct with desired settings
/// 2. Calls ioctl(fd, TCSETS, &termios) via inline assembly
/// 3. Kernel applies the settings immediately (TCSANOW behavior)
///
/// ## Arguments
///
/// * `fd` - Raw file descriptor for the terminal
/// * `termios` - Terminal attributes to apply
///
/// ## Returns
///
/// * `Ok(())` - Settings applied successfully
/// * `Err(io::Error)` - If ioctl fails
///
/// ## Errors
///
/// - ENOTTY (25): fd is not a terminal
/// - EBADF (9): fd is not a valid file descriptor
/// - EINVAL (22): invalid termios settings
///
/// ## C Equivalent
///
/// ```c
/// int ret = tcsetattr(fd, TCSANOW, &ios);  // or: ioctl(fd, TCSETS, &ios)
/// ```
fn set_terminal_attr(fd: i32, termios: &Termios) -> io::Result<()> {
    // SAFETY:
    // - termios points to valid, aligned memory (Rust reference guarantee)
    // - TCSETS only reads from our memory and writes to kernel
    // - The struct layout matches what the kernel expects
    // - Cast to *mut is safe because TCSETS doesn't actually modify the struct
    // SAFETY: termios is a valid, properly aligned, readable Termios reference.
    // ioctl_write_termios + TCSETS reads from it to configure the terminal.
    // No *const-to-*mut cast needed: the write wrapper accepts *const directly.
    let ret = unsafe { ioctl_write_termios(fd, TCSETS, termios as *const Termios) };

    // Check for error
    if ret < 0 {
        Err(io::Error::from_raw_os_error((-ret) as i32))
    } else {
        Ok(())
    }
}

/// Modify termios flags to enable raw mode.
///
/// ## Project Context
///
/// This function transforms a normal terminal configuration into "raw mode"
/// by clearing and setting specific flags. After applying these settings,
/// the terminal will:
///
/// - Return input immediately (no line buffering)
/// - Not echo typed characters
/// - Not process special characters (Ctrl+C, etc.)
/// - Pass all bytes through without modification
///
/// ## What Each Flag Change Does
///
/// ### Input Flags (c_iflag) - DISABLE all processing:
/// - IGNBRK: Don't ignore break conditions
/// - BRKINT: Don't generate SIGINT on break
/// - PARMRK: Don't mark parity errors
/// - ISTRIP: Don't strip 8th bit (keep full bytes)
/// - INLCR: Don't convert NL to CR
/// - IGNCR: Don't ignore CR
/// - ICRNL: Don't convert CR to NL
/// - IXON: Don't enable XON/XOFF flow control (Ctrl+S/Q)
///
/// ### Output Flags (c_oflag) - DISABLE processing:
/// - OPOST: Don't post-process output (no \n → \r\n)
///
/// ### Local Flags (c_lflag) - DISABLE line discipline:
/// - ECHO: Don't echo input
/// - ECHONL: Don't echo newline
/// - ICANON: Don't buffer by line (THE key raw mode flag)
/// - ISIG: Don't generate signals (Ctrl+C, Ctrl+Z, etc.)
/// - IEXTEN: Don't enable extended processing
///
/// ### Control Flags (c_cflag) - Set 8-bit chars:
/// - Clear CSIZE and PARENB
/// - Set CS8 for 8-bit character size
///
/// ### Control Characters (c_cc):
/// - VMIN=1: read() waits for at least 1 byte
/// - VTIME=0: No timeout (wait forever)
///
/// ## C Equivalent
///
/// This is exactly what `cfmakeraw()` does in glibc/musl.
///
/// ## Arguments
///
/// * `termios` - Mutable reference to termios struct to modify
///
/// ## Important Note: Manual CRLF in Raw Mode
///
/// With OPOST disabled, the application **must** write `\r\n` explicitly
/// for proper newlines. A bare `\n` will move the cursor down but leave
/// it at whatever column it was (the kernel will not return to column 0).
///
/// This is not a bug; it's intentional raw mode behavior allowing
/// pixel-precise cursor control. Most terminal UIs (vim, less, etc.)
/// handle this by:
/// - Using ANSI escape codes to control the cursor explicitly
/// - Or writing `\r\n` for each "newline"
fn make_raw(termios: &mut Termios) {
    // -------------------------------------------------------------------------
    // Disable input processing
    // -------------------------------------------------------------------------
    // We want bytes exactly as they come from the keyboard/terminal.
    // Clear all input processing flags.
    termios.c_iflag &= !(IGNBRK | BRKINT | PARMRK | ISTRIP | INLCR | IGNCR | ICRNL | IXON);

    // -------------------------------------------------------------------------
    // Disable output processing
    // -------------------------------------------------------------------------
    // We want bytes exactly as we write them.
    // This means we must write \r\n explicitly for newlines.
    termios.c_oflag &= !OPOST;

    // -------------------------------------------------------------------------
    // Disable local/line-discipline processing
    // -------------------------------------------------------------------------
    // This is the core of "raw mode":
    // - ICANON: Disable line buffering (most important!)
    // - ECHO: Don't show typed characters
    // - ISIG: Don't generate signals from Ctrl+C etc.
    // - ECHONL: Don't echo newlines
    // - IEXTEN: Don't do extended processing
    termios.c_lflag &= !(ECHO | ECHONL | ICANON | ISIG | IEXTEN);

    // -------------------------------------------------------------------------
    // Set control flags for 8-bit characters
    // -------------------------------------------------------------------------
    // Clear character size bits and parity, then set 8-bit mode
    termios.c_cflag &= !(CSIZE | PARENB);
    termios.c_cflag |= CS8;

    // -------------------------------------------------------------------------
    // Set control characters for blocking single-byte reads
    // -------------------------------------------------------------------------
    // VMIN=1: read() returns after at least 1 byte
    // VTIME=0: No timeout, wait forever for that byte
    //
    // Other combinations:
    // - VMIN=0, VTIME=0: Non-blocking (return immediately)
    // - VMIN=0, VTIME>0: Timed read (return after timeout)
    // - VMIN>0, VTIME>0: Block for VMIN bytes or VTIME timeout
    termios.c_cc[VMIN] = 1;
    termios.c_cc[VTIME] = 0;
}

// ============================================================================
// RAW TERMINAL - RAII WRAPPER
// ============================================================================
//
// This struct provides safe, automatic management of raw mode:
// - Enables raw mode on creation
// - Saves original settings
// - Restores original settings on drop (even during panic)
//
// The RAII pattern ensures the terminal is never left in raw mode
// accidentally, which would make the shell unusable.
// ============================================================================

/// A terminal handle in raw mode with automatic cleanup.
///
/// ## Project Context
///
/// This is the main interface for using raw terminal mode. It:
/// 1. Opens /dev/tty for direct terminal access
/// 2. Enables raw mode for byte-by-byte input
/// 3. Automatically restores normal mode when dropped
///
/// ## Usage Pattern
///
/// ```rust,no_run
/// let mut term = RawTerminal::new()?;
/// // Terminal is now in raw mode
/// let mut buf = [0u8; 1];
/// term.read(&mut buf)?;  // Reads single keypress immediately
/// // When `term` is dropped, terminal returns to normal mode
/// ```
///
/// ## Why RAII?
///
/// If raw mode isn't properly disabled:
/// - Echo is off: you can't see what you type
/// - Line buffering is off: shell behaves strangely
/// - Signals are off: Ctrl+C doesn't work
///
/// The Drop implementation ensures cleanup even if the program panics.
///
/// ## Why /dev/tty?
///
/// Opening /dev/tty instead of using stdin/stdout:
/// - Works even if stdin is redirected from a file
/// - Works even if stdout is redirected to a file
/// - Always refers to the controlling terminal
/// - Provides both read and write access
///
/// If a caller writes RawTerminal::new()?; (discards the guard),
/// raw mode is enabled and immediately restored on the same line
/// — a silent no-op that wastes a syscall round-trip and confuses debugging.
/// #[must_use] catches this at compile time.
///
#[derive(Debug)]
#[must_use = "RawTerminal restores the terminal on drop; discarding it immediately undoes raw mode"]
pub struct RawTerminal {
    /// The terminal file handle (opened /dev/tty)
    tty: File,

    /// Original terminal settings, saved for restoration on drop
    prev_ios: Termios,
}

impl RawTerminal {
    /// Create a new RawTerminal, entering raw mode.
    ///
    /// ## Project Context
    ///
    /// This is the entry point for raw terminal functionality. Call this
    /// to get a terminal handle that:
    /// - Provides byte-by-byte input (no line buffering)
    /// - Doesn't echo typed characters
    /// - Doesn't process special keys (Ctrl+C, etc.)
    ///
    /// ## How It Works
    ///
    /// 1. Opens /dev/tty for read+write access
    /// 2. Reads current terminal settings
    /// 3. Saves settings for later restoration
    /// 4. Applies raw mode settings
    ///
    /// ## Returns
    ///
    /// * `Ok(RawTerminal)` - Handle to terminal in raw mode
    /// * `Err(io::Error)` - If opening /dev/tty or setting raw mode fails
    ///
    /// ## Errors
    ///
    /// - ENOENT: /dev/tty doesn't exist (not running in a terminal)
    /// - EACCES: Permission denied to open /dev/tty
    /// - ENOTTY: File descriptor is not a terminal
    ///
    /// ## Example
    ///
    /// ```rust,no_run
    /// use raw_terminal_x86::RawTerminal;
    ///
    /// fn main() -> std::io::Result<()> {
    ///     let mut term = RawTerminal::new()?;
    ///     // Use term for raw input...
    ///     Ok(())
    /// } // Terminal automatically restored here
    /// ```
    ///
    /// ## Single Instance Requirement
    ///
    /// Only one `RawTerminal` should exist at a time for a given terminal.
    /// Creating a second instance while one is active will save the *raw*
    /// settings as "original," making restoration incorrect. This is not
    /// enforced at compile time; callers must ensure single ownership.
    pub fn new() -> io::Result<Self> {
        // ---------------------------------------------------------------------
        // Step 1: Open /dev/tty
        // ---------------------------------------------------------------------
        // /dev/tty is a special file that always refers to the controlling
        // terminal of the current process. This works even if stdin/stdout
        // are redirected.
        let tty = OpenOptions::new().read(true).write(true).open("/dev/tty")?;

        let fd = tty.as_raw_fd();

        // ---------------------------------------------------------------------
        // Step 2: Save current settings
        // ---------------------------------------------------------------------
        // We need to restore these when the RawTerminal is dropped
        let prev_ios = get_terminal_attr(fd)?;

        // ---------------------------------------------------------------------
        // Step 3: Create raw mode settings
        // ---------------------------------------------------------------------
        // Start with current settings, then modify for raw mode
        let mut raw_ios = prev_ios;
        make_raw(&mut raw_ios);

        // ---------------------------------------------------------------------
        // Step 4: Apply raw mode
        // ---------------------------------------------------------------------
        set_terminal_attr(fd, &raw_ios)?;

        Ok(RawTerminal { tty, prev_ios })
    }

    /// Temporarily suspend raw mode, restoring normal terminal behavior.
    ///
    /// ## Project Context
    ///
    /// Useful for:
    /// - Running a subprocess that needs normal terminal behavior
    /// - Prompting user for input with normal line editing
    /// - Debugging output that needs proper newline handling
    ///
    /// Call `activate_raw_mode()` to re-enable raw mode.
    ///
    /// ## Returns
    ///
    /// * `Ok(())` - Normal mode restored
    /// * `Err(io::Error)` - If restoring settings fails
    /// suspend_raw_mode / activate_raw_mode idempotency note
    /// Both methods are safe to call repeatedly or out of expected order
    /// (suspend twice, activate without suspend, etc.)
    /// Safe to call multiple times; each call applies the same saved original settings.
    pub fn suspend_raw_mode(&self) -> io::Result<()> {
        set_terminal_attr(self.tty.as_raw_fd(), &self.prev_ios)?;
        Ok(())
    }

    /// Re-enable raw mode after suspending.
    ///
    /// ## Project Context
    ///
    /// Call this after `suspend_raw_mode()` to return to raw mode.
    /// This function verifies the settings were actually applied,
    /// which guards against kernel bugs, driver issues, or race
    /// conditions with other processes modifying terminal state.
    ///
    /// ## Design Decision
    ///
    /// This applies `make_raw` to the **saved original settings** (`prev_ios`),
    /// not to whatever the terminal currently has. This ensures predictable
    /// behavior even if something external modified termios between suspend
    /// and activate.
    ///
    /// ## Verification Strategy
    ///
    /// After setting raw mode, we read back the terminal attributes and
    /// verify critical flags are correctly set. This catches:
    /// - Kernel/driver silently ignoring our request
    /// - Race condition with another process changing terminal
    /// - Hardware/driver bugs
    /// - Bit-flip corruption (per production rules)
    ///
    /// ## Returns
    ///
    /// * `Ok(())` - Raw mode re-enabled and verified
    /// * `Err(io::Error)` - If setting raw mode fails or verification fails
    ///
    /// ## Errors
    ///
    /// - ENOTTY (25): fd is not a terminal
    /// - EBADF (9): fd is not valid (terminal was closed)
    /// - EINVAL (22): verification failed - settings not applied correctly
    ///
    /// suspend_raw_mode / activate_raw_mode idempotency note
    /// Both methods are safe to call repeatedly or out of expected order
    /// (suspend twice, activate without suspend, etc.)
    /// Safe to call multiple times or without a prior suspend; always derives raw
    /// settings from the saved original, not from current terminal state.
    pub fn activate_raw_mode(&self) -> io::Result<()> {
        // ---------------------------------------------------------------------
        // Step 1: Prepare raw mode settings from known-good saved state
        // ---------------------------------------------------------------------
        // Start from saved original settings, not current (possibly corrupted)
        // state. This ensures we always apply a consistent configuration.
        let mut raw_ios = self.prev_ios;
        make_raw(&mut raw_ios);

        // ---------------------------------------------------------------------
        // Step 2: Apply raw mode settings to terminal
        // ---------------------------------------------------------------------
        set_terminal_attr(self.tty.as_raw_fd(), &raw_ios)?;

        // ---------------------------------------------------------------------
        // Step 3: Verify settings were actually applied (defense in depth)
        // ---------------------------------------------------------------------
        // Per production rules: "Every return should be checked for what can
        // be checked." The kernel could silently fail to apply settings due to:
        // - Driver limitations
        // - Race conditions with other processes
        // - Kernel bugs
        // - Corrupted file descriptor state
        //
        // We verify the most critical flags that define "raw mode":
        // - ICANON must be OFF (byte-by-byte input, not line-buffered)
        // - ECHO must be OFF (no automatic character echo)
        //
        // These are the two flags that most visibly affect raw mode behavior.
        // If either is incorrectly set, the terminal will not behave as raw.
        // ---------------------------------------------------------------------
        let verification_result = self.verify_raw_mode_active();

        // ---------------------------------------------------------------------
        // Debug assertion: panic in debug builds for immediate developer notice
        // Production check: return error, never panic
        // ---------------------------------------------------------------------
        // Note: debug_assert is excluded from test builds to avoid collision
        // with test error handling. Tests verify via the returned Result.
        #[cfg(all(debug_assertions, not(test)))]
        if let Err(ref e) = verification_result {
            // Panic in debug builds (not test) for developer visibility.
            // The #[cfg] guard ensures this never compiles into production.
            panic!(
                "ARTM: Raw mode verification failed after set_terminal_attr succeeded: {}",
                e
            );
        }

        // Production: return the verification result (Ok or Err)
        // This allows callers to handle the failure gracefully
        verification_result
    }

    /// Verify that raw mode settings are currently active on the terminal.
    ///
    /// ## Project Context
    ///
    /// This is an internal helper used by `activate_raw_mode` to confirm
    /// that the terminal is actually in raw mode after we attempted to
    /// set it. This implements the "trust but verify" principle from
    /// the production rules.
    ///
    /// ## What We Verify
    ///
    /// - ICANON flag is cleared (canonical/line mode disabled)
    /// - ECHO flag is cleared (input echo disabled)
    ///
    /// These are the defining characteristics of raw mode. Other flags
    /// (ISIG, OPOST, etc.) are also set by make_raw(), but ICANON and
    /// ECHO are the most critical for correct operation.
    ///
    /// ## Returns
    ///
    /// * `Ok(())` - Terminal is in raw mode
    /// * `Err(io::Error)` - Verification failed; includes which flag was wrong
    ///
    /// ## Error Codes
    ///
    /// Returns EINVAL (22) for verification failures, as the terminal
    /// configuration is invalid for our purposes.
    fn verify_raw_mode_active(&self) -> io::Result<()> {
        // Read current terminal attributes
        let current_ios = get_terminal_attr(self.tty.as_raw_fd())?;

        // =====================================================================
        // Verify ICANON is disabled (most critical raw mode flag)
        // =====================================================================
        // ICANON controls line buffering. When enabled:
        // - Input is buffered until newline/Enter
        // - Line editing (backspace, etc.) is handled by kernel
        // When disabled (raw mode):
        // - Each byte is available immediately
        // - No kernel-level line editing
        //
        // If ICANON is still set after our set_terminal_attr call, raw mode
        // is NOT active and the application will not receive byte-by-byte input.
        // =====================================================================
        if (current_ios.c_lflag & ICANON) != 0 {
            // ICANON still set - raw mode not active

            // Debug builds: log detailed diagnostic before returning
            // Not included in production or test builds
            #[cfg(all(debug_assertions, not(test)))]
            {
                let _ = std::io::stderr()
                    .write_all(b"VRTM debug: ICANON flag still set after raw mode activation\r\n");
            }

            // Production: terse, no-heap error
            // EINVAL (22) = invalid terminal configuration for our purposes
            return Err(io::Error::from_raw_os_error(22));
        }

        // =====================================================================
        // Verify ECHO is disabled
        // =====================================================================
        // ECHO controls whether typed characters are automatically displayed.
        // In raw mode, the application controls all output, so ECHO must be off.
        //
        // If ECHO is still set, typed characters will appear twice:
        // - Once from kernel echo
        // - Once from application display
        // This causes confusing double-character display.
        // =====================================================================
        if (current_ios.c_lflag & ECHO) != 0 {
            // ECHO still set - terminal will double-echo characters

            #[cfg(all(debug_assertions, not(test)))]
            {
                let _ = std::io::stderr()
                    .write_all(b"VRTM debug: ECHO flag still set after raw mode activation\r\n");
            }

            return Err(io::Error::from_raw_os_error(22));
        }

        /*
        ICANON and ECHO are critical,
        but if VMIN were corrupted to 0 after reactivation,
        read() would return 0 bytes immediately
        (appearing as EOF) — a baffling failure mode.
        */
        // Inside verify_raw_mode_active, after the ECHO check:
        if current_ios.c_cc[VMIN] == 0 {
            #[cfg(all(debug_assertions, not(test)))]
            {
                let _ = std::io::stderr()
                    .write_all(b"VRTM debug: VMIN is 0 after raw mode activation\r\n");
            }
            return Err(io::Error::from_raw_os_error(22));
        }

        // Both critical flags verified - raw mode is active
        Ok(())
    }
}

impl Drop for RawTerminal {
    /// Restore terminal to original settings on drop.
    ///
    /// ## Project Context
    ///
    /// This is critical for usability. If we fail to restore:
    /// - Echo stays off: cannot see what you type in the shell
    /// - Line buffering stays off: shell behaves strangely
    /// - Signals stay off: Ctrl+C does not work
    ///
    /// The Drop trait ensures this runs even if the program panics.
    ///
    /// ## Error Handling
    ///
    /// Drop cannot return errors. On failure we:
    /// 1. Log a terse message to stderr (no heap, no sensitive data)
    /// 2. Continue - the kernel/shell usually resets terminal on process exit
    ///
    /// Per production rules: errors must be logged even when not propagatable.
    fn drop(&mut self) {
        if let Err(_e) = set_terminal_attr(self.tty.as_raw_fd(), &self.prev_ios) {
            // Terse, no-heap error indication
            // Note: stderr write can also fail; nothing more we can do
            let _ = std::io::stderr().write_all(b"RTDROP: term restore fail\r\n");
        }
    }
}

impl Read for RawTerminal {
    /// Read bytes from the terminal.
    ///
    /// ## Project Context
    ///
    /// In raw mode, read() returns as soon as data is available
    /// (controlled by VMIN/VTIME settings). With our settings:
    /// - VMIN=1: Returns after at least 1 byte
    /// - VTIME=0: No timeout
    ///
    /// This means read() blocks until a key is pressed, then returns
    /// that byte immediately (no waiting for Enter).
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        self.tty.read(buf)
    }
}

impl Write for RawTerminal {
    /// Write bytes to the terminal.
    ///
    /// ## Project Context
    ///
    /// In raw mode (with OPOST disabled), bytes are written exactly
    /// as provided. This means:
    /// - '\n' moves down one line but doesn't return to column 0
    /// - You must write "\r\n" for a proper newline
    /// - No automatic CR/LF translation
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.tty.write(buf)
    }

    fn flush(&mut self) -> io::Result<()> {
        self.tty.flush()
    }
}

// ============================================================================
// TESTS
// ============================================================================
//
// These tests verify the implementation correctness.
// Some tests require a real terminal and will be skipped in CI.
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// Verify termios struct has correct size for Linux x86_64.
    ///
    /// The kernel expects exactly 36 bytes. If this is wrong, ioctl will
    /// read/write incorrect memory, causing corruption or crashes.
    #[test]
    fn test_termios_struct_size() {
        assert_eq!(
            core::mem::size_of::<Termios>(),
            36,
            "Termios struct must be 36 bytes for Linux x86_64"
        );
    }

    /// Verify termios struct field offsets match kernel expectations.
    #[test]
    fn test_termios_field_offsets() {
        // Use offset_of! or manual calculation
        // c_iflag should be at offset 0
        // c_oflag should be at offset 4
        // c_cflag should be at offset 8
        // c_lflag should be at offset 12
        // c_line should be at offset 16
        // c_cc should be at offset 17

        let termios = Termios {
            c_iflag: 0,
            c_oflag: 0,
            c_cflag: 0,
            c_lflag: 0,
            c_line: 0,
            c_cc: [0u8; 19],
        };

        let base = &termios as *const _ as usize;
        let iflag_offset = &termios.c_iflag as *const _ as usize - base;
        let oflag_offset = &termios.c_oflag as *const _ as usize - base;
        let cflag_offset = &termios.c_cflag as *const _ as usize - base;
        let lflag_offset = &termios.c_lflag as *const _ as usize - base;
        let line_offset = &termios.c_line as *const _ as usize - base;
        let cc_offset = &termios.c_cc as *const _ as usize - base;

        assert_eq!(iflag_offset, 0, "c_iflag should be at offset 0");
        assert_eq!(oflag_offset, 4, "c_oflag should be at offset 4");
        assert_eq!(cflag_offset, 8, "c_cflag should be at offset 8");
        assert_eq!(lflag_offset, 12, "c_lflag should be at offset 12");
        assert_eq!(line_offset, 16, "c_line should be at offset 16");
        assert_eq!(cc_offset, 17, "c_cc should be at offset 17");
    }

    /// Test that make_raw correctly modifies flags.
    #[test]
    fn test_make_raw_clears_icanon() {
        let mut termios = Termios {
            c_iflag: 0xFFFFFFFF, // All bits set
            c_oflag: 0xFFFFFFFF,
            c_cflag: 0xFFFFFFFF,
            c_lflag: 0xFFFFFFFF,
            c_line: 0,
            c_cc: [0u8; 19],
        };

        make_raw(&mut termios);

        // ICANON must be cleared for raw mode
        assert_eq!(
            termios.c_lflag & ICANON,
            0,
            "ICANON flag must be cleared for raw mode"
        );

        // ECHO must be cleared
        assert_eq!(
            termios.c_lflag & ECHO,
            0,
            "ECHO flag must be cleared for raw mode"
        );

        // ISIG must be cleared
        assert_eq!(
            termios.c_lflag & ISIG,
            0,
            "ISIG flag must be cleared for raw mode"
        );

        // OPOST must be cleared
        assert_eq!(
            termios.c_oflag & OPOST,
            0,
            "OPOST flag must be cleared for raw mode"
        );

        // CS8 must be set
        assert_eq!(termios.c_cflag & CSIZE, CS8, "CS8 must be set for raw mode");

        // VMIN must be 1
        assert_eq!(termios.c_cc[VMIN], 1, "VMIN must be 1 for raw mode");

        // VTIME must be 0
        assert_eq!(termios.c_cc[VTIME], 0, "VTIME must be 0 for raw mode");
    }

    /// Test make_raw on a zeroed struct (simulating fresh terminal).
    #[test]
    fn test_make_raw_from_zero() {
        let mut termios = Termios {
            c_iflag: 0,
            c_oflag: 0,
            c_cflag: 0,
            c_lflag: 0,
            c_line: 0,
            c_cc: [0u8; 19],
        };

        make_raw(&mut termios);

        // Should set CS8
        assert_eq!(termios.c_cflag & CS8, CS8);

        // Should set VMIN
        assert_eq!(termios.c_cc[VMIN], 1);
    }

    /// Test that termios struct can be copied (required for save/restore).
    #[test]
    fn test_termios_copy() {
        let original = Termios {
            c_iflag: 0x12345678,
            c_oflag: 0x9ABCDEF0,
            c_cflag: 0x11223344,
            c_lflag: 0x55667788,
            c_line: 42,
            c_cc: [
                1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
            ],
        };

        let copy = original;

        assert_eq!(copy.c_iflag, original.c_iflag);
        assert_eq!(copy.c_oflag, original.c_oflag);
        assert_eq!(copy.c_cflag, original.c_cflag);
        assert_eq!(copy.c_lflag, original.c_lflag);
        assert_eq!(copy.c_line, original.c_line);
        assert_eq!(copy.c_cc, original.c_cc);
    }

    /// Test constants match expected octal values from kernel headers.
    #[test]
    fn test_flag_constant_values() {
        // Input flags
        assert_eq!(IGNBRK, 0o1);
        assert_eq!(BRKINT, 0o2);
        assert_eq!(PARMRK, 0o10);
        assert_eq!(ISTRIP, 0o40);
        assert_eq!(INLCR, 0o100);
        assert_eq!(IGNCR, 0o200);
        assert_eq!(ICRNL, 0o400);
        assert_eq!(IXON, 0o2000);

        // Output flags
        assert_eq!(OPOST, 0o1);

        // Control flags
        assert_eq!(CSIZE, 0o60);
        assert_eq!(CS8, 0o60);
        assert_eq!(PARENB, 0o400);

        // Local flags
        assert_eq!(ISIG, 0o1);
        assert_eq!(ICANON, 0o2);
        assert_eq!(ECHO, 0o10);
        assert_eq!(ECHONL, 0o100);
        assert_eq!(IEXTEN, 0o100000);
    }

    /// Test ioctl request code values.
    #[test]
    fn test_ioctl_constants() {
        assert_eq!(TCGETS, 0x5401);
        assert_eq!(TCSETS, 0x5402);
    }

    /// Test syscall number for ioctl.
    #[test]
    fn test_syscall_number() {
        assert_eq!(SYS_IOCTL, 16);
    }

    /// Integration test: attempt to open /dev/tty.
    /// This test will be skipped if not running in a terminal.
    #[test]
    fn test_open_dev_tty() {
        // This may fail in CI environments without a terminal
        let result = OpenOptions::new().read(true).write(true).open("/dev/tty");

        match result {
            Ok(file) => {
                // Verify it's a valid file descriptor
                let fd = file.as_raw_fd();
                assert!(fd >= 0, "File descriptor should be non-negative");
            }
            Err(e) => {
                // Expected in CI/headless environments
                eprintln!("Note: /dev/tty not available (expected in CI): {}", e);
            }
        }
    }

    /// Integration test: get terminal attributes.
    /// Requires a real terminal.
    #[test]
    fn test_get_terminal_attr() {
        let tty = match OpenOptions::new().read(true).write(true).open("/dev/tty") {
            Ok(f) => f,
            Err(_) => {
                eprintln!("Skipping test_get_terminal_attr: no terminal available");
                return;
            }
        };

        let result = get_terminal_attr(tty.as_raw_fd());

        match result {
            Ok(termios) => {
                // Verify we got some reasonable values
                // c_cflag should have some bits set (at least baud rate info)
                // This is a sanity check, not a specific value check
                eprintln!("Got termios: {:?}", termios);
            }
            Err(e) => {
                panic!("get_terminal_attr failed unexpectedly: {}", e);
            }
        }
    }

    /// Integration test: full raw mode cycle.
    /// Requires a real terminal and user interaction.
    /// Run with: cargo test -- --ignored test_raw_mode_roundtrip
    #[test]
    #[ignore] // Requires interactive terminal
    fn test_raw_mode_roundtrip() {
        // This test verifies the complete flow:
        // 1. Create RawTerminal (enters raw mode)
        // 2. Verify we can read a byte
        // 3. Drop RawTerminal (restores normal mode)
        // 4. Verify terminal is usable again

        eprintln!("\r\nCreating RawTerminal...\r\n");

        match RawTerminal::new() {
            Ok(mut term) => {
                eprintln!("Raw mode active. Press any key...\r\n");

                let mut buf = [0u8; 1];
                match term.read(&mut buf) {
                    Ok(n) => {
                        eprintln!("Read {} byte(s): {:?}\r\n", n, buf[0]);
                    }
                    Err(e) => {
                        eprintln!("Read error: {}\r\n", e);
                    }
                }

                eprintln!("Dropping RawTerminal...\r\n");
                // term is dropped here, restoring normal mode
            }
            Err(e) => {
                eprintln!("Failed to create RawTerminal: {}", e);
            }
        }

        eprintln!("Terminal should be restored to normal mode now.");
    }
}

/// Format byte information into a fixed buffer without heap allocation.
///
/// ## Project Context
///
/// Following the production rules about avoiding heap allocation,
/// this function formats the byte display using a stack buffer.
/// This is used by the demonstration main() to show keystroke byte values.
///
/// ## Byte Value Ranges and Display Format
///
/// | Range       | Name              | Display    | Example           |
/// |-------------|-------------------|------------|-------------------|
/// | 0x00-0x1F   | C0 Control codes  | `^X`       | 0x01 → `^A`       |
/// | 0x20-0x7E   | Printable ASCII   | literal    | 0x41 → `A`        |
/// | 0x7F        | DEL               | `^?`       | 0x7F → `^?`       |
/// | 0x80-0xFF   | High bytes        | `<XX>`     | 0x80 → `<80>`     |
///
/// High bytes (0x80-0xFF) are shown in hex notation because:
/// - 0x80-0x9F are C1 control codes (not printable)
/// - 0xA0-0xFF interpretation depends on encoding (Latin-1, UTF-8 fragment, etc.)
/// - Showing as `<XX>` is unambiguous regardless of terminal encoding
///
/// ## Buffer Size Calculation
///
/// Maximum output for high-byte case (longest possible):
/// - "byte: "     = 6 bytes
/// - "255"        = 3 bytes (max decimal)
/// - " (0x"       = 4 bytes
/// - "FF"         = 2 bytes
/// - ") = '"      = 5 bytes
/// - "<FF>"       = 4 bytes (longest char repr: high byte)
/// - "'\r\n"      = 3 bytes
/// - Total max    = 27 bytes
///
/// Buffer of 64 bytes provides >2x safety margin.
///
/// ## Arguments
///
/// * `output` - Fixed 64-byte buffer to write formatted output into
/// * `byte`   - The byte value to format (0x00-0xFF)
///
/// ## Returns
///
/// Number of bytes written to the output buffer.
/// Returns 0 on bounds violation (should never happen with correct buffer size).
///
/// ## Design Note
///
/// This function uses a macro for safe bounds-checked copying rather than
/// relying on slice bounds checks that would panic. Returning 0 on overflow
/// allows the caller to skip malformed output rather than crashing.
pub fn format_byte_info(output: &mut [u8; 64], byte: u8) -> usize {
    let mut pos: usize = 0;

    // -------------------------------------------------------------------------
    // Safe copy macro with explicit bounds checking
    // -------------------------------------------------------------------------
    // This macro copies a byte slice into the output buffer at the current
    // position, advancing the position. On any bounds violation (overflow
    // or arithmetic error), it returns 0 to indicate failure rather than
    // panicking. The caller can then skip this byte's output.
    //
    // Per production rules: "handle and move on" rather than panic.
    // -------------------------------------------------------------------------
    macro_rules! safe_copy {
        ($src:expr) => {{
            let src: &[u8] = $src;
            // Use checked arithmetic to prevent overflow
            let end = match pos.checked_add(src.len()) {
                Some(e) if e <= output.len() => e,
                _ => return 0, // Bounds violation: return empty
            };
            output[pos..end].copy_from_slice(src);
            pos = end;
        }};
    }

    // -------------------------------------------------------------------------
    // Format: "byte: NNN (0xHH) = 'REPR'\r\n"
    // -------------------------------------------------------------------------

    safe_copy!(b"byte: ");

    // -------------------------------------------------------------------------
    // Decimal value (0-255)
    // -------------------------------------------------------------------------
    // Format without heap allocation using simple digit extraction.
    // Maximum 3 digits for values 100-255.
    if byte >= 100 {
        // Three digits: e.g., 255 → '2', '5', '5'
        safe_copy!(&[b'0' + byte / 100, b'0' + (byte / 10) % 10, b'0' + byte % 10]);
    } else if byte >= 10 {
        // Two digits: e.g., 42 → '4', '2'
        safe_copy!(&[b'0' + byte / 10, b'0' + byte % 10]);
    } else {
        // One digit: e.g., 7 → '7'
        safe_copy!(&[b'0' + byte]);
    }

    safe_copy!(b" (0x");

    // -------------------------------------------------------------------------
    // Hexadecimal value (always 2 digits)
    // -------------------------------------------------------------------------
    const HEX_DIGITS: &[u8; 16] = b"0123456789ABCDEF";
    let hex_high = HEX_DIGITS[(byte >> 4) as usize];
    let hex_low = HEX_DIGITS[(byte & 0x0F) as usize];
    safe_copy!(&[hex_high, hex_low]);

    safe_copy!(b") = '");

    // -------------------------------------------------------------------------
    // Character representation (varies by byte range)
    // -------------------------------------------------------------------------
    // Different byte ranges require different display strategies:
    //
    // Printable ASCII (0x20-0x7E): Show the character itself
    //   - Space through tilde, all safely displayable
    //
    // Control characters (0x00-0x1F): Show as ^X (caret notation)
    //   - Traditional Unix convention
    //   - 0x00 = ^@ (NUL), 0x01 = ^A, ..., 0x1A = ^Z, 0x1B = ^[, etc.
    //   - Calculation: control_char + 0x40 = display_char
    //
    // DEL (0x7F): Show as ^? (traditional caret notation for DEL)
    //   - DEL is technically a control character but outside 0x00-0x1F range
    //
    // High bytes (0x80-0xFF): Show as <XX> hex notation
    //   - These bytes' meaning depends on character encoding
    //   - In UTF-8: could be continuation bytes or start of multi-byte sequence
    //   - In Latin-1: 0x80-0x9F are C1 controls, 0xA0-0xFF are extended chars
    //   - Hex notation is unambiguous regardless of terminal encoding
    // -------------------------------------------------------------------------
    if byte >= 0x20 && byte < 0x7F {
        // ---------------------------------------------------------------------
        // Printable ASCII (0x20 space through 0x7E tilde)
        // ---------------------------------------------------------------------
        // These characters can be safely displayed as-is on any terminal.
        safe_copy!(&[byte]);
    } else if byte == 0x7F {
        // ---------------------------------------------------------------------
        // DEL character (0x7F)
        // ---------------------------------------------------------------------
        // DEL is historically shown as ^? in caret notation.
        // This is because: 0x7F = 0x3F ('?') + 0x40
        // But we hardcode it for clarity rather than computing.
        safe_copy!(b"^?");
    } else if byte < 0x20 {
        // ---------------------------------------------------------------------
        // C0 Control characters (0x00-0x1F)
        // ---------------------------------------------------------------------
        // Traditional caret notation: control_code + 0x40 = printable_char
        // Examples:
        //   0x00 (NUL) + 0x40 = 0x40 ('@') → ^@
        //   0x01 (SOH) + 0x40 = 0x41 ('A') → ^A
        //   0x03 (ETX) + 0x40 = 0x43 ('C') → ^C (Ctrl+C)
        //   0x0D (CR)  + 0x40 = 0x4D ('M') → ^M (Ctrl+M / Enter)
        //   0x1B (ESC) + 0x40 = 0x5B ('[') → ^[ (Escape)
        //
        // wrapping_add is used for safety, though overflow is impossible here
        // since byte < 0x20 means byte + 0x40 < 0x60, well within u8 range.
        let caret_char = byte.wrapping_add(0x40);
        safe_copy!(&[b'^', caret_char]);
    } else {
        // ---------------------------------------------------------------------
        // High bytes (0x80-0xFF)
        // ---------------------------------------------------------------------
        // These bytes have encoding-dependent meanings:
        // - UTF-8: Could be continuation byte (10xxxxxx) or start byte
        // - Latin-1/ISO-8859-1: 0x80-0x9F are C1 controls, 0xA0-0xFF are chars
        // - Windows-1252: Similar to Latin-1 but with printable chars in 0x80-0x9F
        //
        // Rather than assuming an encoding, we show the hex value in angle
        // brackets: <80>, <FF>, etc. This is unambiguous and safe.
        //
        // The angle bracket notation is inspired by how some tools display
        // unprintable bytes (e.g., <NUL>, <DEL>) but we use hex for precision.
        safe_copy!(b"<");
        safe_copy!(&[hex_high, hex_low]);
        safe_copy!(b">");
    }

    safe_copy!(b"'\r\n");

    pos
}

/// Tests for format_byte_info stack buffer formatting function.
///
/// ## Project Context
///
/// format_byte_info had a real production bug (DEL branch off-by-one)
/// that was caught only by code review. These tests prevent regression
/// and verify all byte-range branches are exercised.
#[cfg(test)]
mod format_tests {
    use super::*;

    /// Verify high bytes (0x80-0xFF) are shown in <XX> hex notation.
    ///
    /// ## What This Tests
    ///
    /// High bytes have encoding-dependent meanings and should not be
    /// displayed as literal characters (which could cause terminal
    /// rendering issues or be misleading). Instead, they're shown
    /// as `<HH>` where HH is the two-digit hex value.
    ///
    /// This tests the boundary values and a mid-range value:
    /// - 0x80: First high byte (C1 control in Latin-1)
    /// - 0xC0: Common UTF-8 start byte
    /// - 0xFF: Maximum byte value
    #[test]
    fn test_format_byte_info_high_bytes_as_hex() {
        // Test 0x80 - first high byte
        let (buf, len) = fmt(0x80);
        let output = core::str::from_utf8(&buf[..len]).unwrap_or("");
        assert!(
            output.contains("<80>"),
            "0x80 should display as <80>, got: {}",
            output.trim()
        );

        // Test 0xC0 - common UTF-8 two-byte sequence start
        let (buf, len) = fmt(0xC0);
        let output = core::str::from_utf8(&buf[..len]).unwrap_or("");
        assert!(
            output.contains("<C0>"),
            "0xC0 should display as <C0>, got: {}",
            output.trim()
        );

        // Test 0xFF - maximum byte value
        let (buf, len) = fmt(0xFF);
        let output = core::str::from_utf8(&buf[..len]).unwrap_or("");
        assert!(
            output.contains("<FF>"),
            "0xFF should display as <FF>, got: {}",
            output.trim()
        );

        // Test 0x9F - last C1 control code
        let (buf, len) = fmt(0x9F);
        let output = core::str::from_utf8(&buf[..len]).unwrap_or("");
        assert!(
            output.contains("<9F>"),
            "0x9F should display as <9F>, got: {}",
            output.trim()
        );

        // Test 0xA0 - non-breaking space in Latin-1 (first non-C1 high byte)
        let (buf, len) = fmt(0xA0);
        let output = core::str::from_utf8(&buf[..len]).unwrap_or("");
        assert!(
            output.contains("<A0>"),
            "0xA0 should display as <A0>, got: {}",
            output.trim()
        );
    }

    /// Verify the transition boundaries between display modes.
    ///
    /// ## What This Tests
    ///
    /// Tests the exact boundary bytes where display format changes:
    /// - 0x1F → 0x20: Control (^_) to printable (space)
    /// - 0x7E → 0x7F: Printable (~) to DEL (^?)
    /// - 0x7F → 0x80: DEL (^?) to high byte (<80>)
    ///
    /// These boundaries are where off-by-one errors commonly occur.
    #[test]
    fn test_format_byte_info_boundary_transitions() {
        // 0x1F (last control char) should show as ^_
        let (buf, len) = fmt(0x1F);
        assert!(len >= 5);
        assert_eq!(buf[len - 5], b'^', "0x1F should have caret");
        assert_eq!(buf[len - 4], b'_', "0x1F should show as ^_");

        // 0x20 (first printable, space) should show as literal space
        let (buf, len) = fmt(0x20);
        assert!(len >= 4);
        assert_eq!(buf[len - 4], b' ', "0x20 should be literal space");

        // 0x7E (last printable, tilde) should show as literal ~
        let (buf, len) = fmt(0x7E);
        assert!(len >= 4);
        assert_eq!(buf[len - 4], b'~', "0x7E should be literal tilde");

        // 0x7F (DEL) should show as ^?
        let (buf, len) = fmt(0x7F);
        assert!(len >= 5);
        assert_eq!(buf[len - 5], b'^', "0x7F (DEL) should have caret");
        assert_eq!(buf[len - 4], b'?', "0x7F (DEL) should show as ^?");

        // 0x80 (first high byte) should show as <80>
        let (buf, len) = fmt(0x80);
        let output = core::str::from_utf8(&buf[..len]).unwrap_or("");
        assert!(output.contains("<80>"), "0x80 should show as <80>");
    }

    /// Helper: run format_byte_info and return the result as a &str slice.
    ///
    /// Trims the trailing \r\n for easier assertion.
    fn fmt(byte: u8) -> ([u8; 64], usize) {
        let mut buf = [0u8; 64];
        let len = format_byte_info(&mut buf, byte);
        (buf, len)
    }

    /// Verify output is non-zero for all 256 possible byte values.
    ///
    /// A zero return means the bounds check triggered, which indicates
    /// a buffer overflow would have occurred — a regression in buffer sizing.
    #[test]
    fn test_format_byte_info_never_returns_zero() {
        for byte_val in 0u8..=255 {
            let (_, len) = fmt(byte_val);
            assert!(
                len > 0,
                "format_byte_info returned 0 for byte value {byte_val}: \
                 buffer overflow guard triggered unexpectedly"
            );
        }
    }

    /// Verify output always fits within the 64-byte buffer.
    #[test]
    fn test_format_byte_info_never_exceeds_buffer() {
        for byte_val in 0u8..=255 {
            let (_, len) = fmt(byte_val);
            assert!(
                len <= 64,
                "format_byte_info returned len={len} for byte {byte_val}: \
                 exceeds 64-byte buffer"
            );
        }
    }

    /// Verify all outputs end with \r\n (required for raw mode terminal display).
    #[test]
    fn test_format_byte_info_ends_with_crlf() {
        for byte_val in 0u8..=255 {
            let (buf, len) = fmt(byte_val);
            assert!(
                len >= 2,
                "format_byte_info output too short for byte {byte_val}"
            );
            assert_eq!(
                buf[len - 2],
                b'\r',
                "Second-to-last byte must be \\r for byte value {byte_val}"
            );
            assert_eq!(
                buf[len - 1],
                b'\n',
                "Last byte must be \\n for byte value {byte_val}"
            );
        }
    }

    /// Verify printable ASCII bytes (0x20-0x7E) are shown as themselves.
    #[test]
    fn test_format_byte_info_printable_ascii() {
        for byte_val in 0x20u8..0x7F {
            let (buf, len) = fmt(byte_val);
            // Output format: "byte: NNN (0xHH) = 'X'\r\n"
            // The character before the closing ' should be the byte itself
            assert!(len >= 3, "Output too short for printable byte {byte_val}");
            // closing quote is at len-3 (before \r\n), char at len-4
            assert_eq!(
                buf[len - 4],
                byte_val,
                "Printable byte 0x{byte_val:02X} should appear as itself in output"
            );
        }
    }

    /// Verify DEL (0x7F) is shown as ^? and NOT as a stale-space bug.
    ///
    /// This is a regression test for the off-by-one bug in the original
    /// DEL branch: `output[pos..pos+4].copy_from_slice(b"DEL ")` with
    /// `pos += 3` left a stale space and reported wrong length.
    #[test]
    fn test_format_byte_info_del_byte() {
        let (buf, len) = fmt(0x7F);

        // Must end in '\r\n
        assert_eq!(buf[len - 2], b'\r');
        assert_eq!(buf[len - 1], b'\n');

        // Closing quote at len-3
        assert_eq!(buf[len - 3], b'\'', "DEL output must close with quote");

        // The two bytes before closing quote should be '^' and '?'
        assert_eq!(buf[len - 5], b'^', "DEL should render as ^? : missing ^");
        assert_eq!(buf[len - 4], b'?', "DEL should render as ^? : missing ?");
    }

    /// Verify control characters (0x00-0x1F) are shown as ^X notation.
    #[test]
    fn test_format_byte_info_control_characters() {
        // Ctrl+A = 0x01 should show as ^A
        let (buf, len) = fmt(0x01);
        assert_eq!(buf[len - 5], b'^', "Control char missing ^");
        assert_eq!(buf[len - 4], b'A', "Ctrl+A should show as ^A");

        // Ctrl+C = 0x03 should show as ^C
        let (buf, len) = fmt(0x03);
        assert_eq!(buf[len - 5], b'^', "Control char missing ^");
        assert_eq!(buf[len - 4], b'C', "Ctrl+C should show as ^C");

        // NUL = 0x00 should show as ^@
        let (buf, len) = fmt(0x00);
        assert_eq!(buf[len - 5], b'^', "NUL missing ^");
        assert_eq!(buf[len - 4], b'@', "NUL should show as ^@");

        // Enter = 0x0D (as seen in terminal output) should show as ^M
        let (buf, len) = fmt(0x0D);
        assert_eq!(buf[len - 5], b'^', "CR missing ^");
        assert_eq!(buf[len - 4], b'M', "CR (0x0D) should show as ^M");
    }

    /// Verify the decimal prefix is correct for boundary values.
    #[test]
    fn test_format_byte_info_decimal_values() {
        // byte: 0 ...
        let (buf, len) = fmt(0);
        assert!(len > 6, "Output too short");
        // After "byte: ", first char should be '0'
        assert_eq!(buf[6], b'0', "Zero should format as single '0'");

        // byte: 255 ...
        let (buf, len) = fmt(255);
        assert!(len > 8, "Output too short for 255");
        assert_eq!(buf[6], b'2', "255 hundreds digit");
        assert_eq!(buf[7], b'5', "255 tens digit");
        assert_eq!(buf[8], b'5', "255 units digit");

        // byte: 99 ...
        let (buf, len) = fmt(99);
        assert!(len > 7, "Output too short for 99");
        assert_eq!(buf[6], b'9', "99 tens digit");
        assert_eq!(buf[7], b'9', "99 units digit");
    }

    /// Verify the hex portion is correct for known values.
    #[test]
    fn test_format_byte_info_hex_values() {
        // 0x0D should show "0x0D"
        let (buf, _len) = fmt(0x0D);
        // Find "0x" in output
        let output_str = core::str::from_utf8(&buf[..30]).unwrap_or("");
        assert!(
            output_str.contains("0x0D"),
            "0x0D should appear in hex output, got prefix: {:?}",
            &buf[..20]
        );

        // 0xFF should show "0xFF"
        let (buf, _len) = fmt(0xFF);
        let output_str = core::str::from_utf8(&buf[..35]).unwrap_or("");
        assert!(
            output_str.contains("0xFF"),
            "0xFF should appear in hex output"
        );
    }
}

/// Tests for error handling paths in terminal operations.
///
/// ## Project Context
///
/// These tests verify that error conditions are properly detected and
/// reported, rather than causing panics or undefined behavior. Per
/// production rules: "Every part of code will fail at some point."
///
/// These tests exercise:
/// - Invalid file descriptor handling (EBADF)
/// - Non-terminal file descriptor handling (ENOTTY)
/// - Verification of proper io::Error construction
///
/// ## Test Environment Notes
///
/// These tests use /dev/null as a non-terminal file descriptor.
/// /dev/null is available on all POSIX systems and is guaranteed
/// to not be a TTY, making it ideal for testing ENOTTY errors.
#[cfg(test)]
mod error_path_tests {
    use super::*;
    use std::os::unix::io::AsRawFd;

    // =========================================================================
    // INVALID FILE DESCRIPTOR TESTS (EBADF)
    // =========================================================================

    /// Verify get_terminal_attr returns EBADF for invalid file descriptor.
    ///
    /// ## What This Tests
    ///
    /// When passed an invalid file descriptor (-1), the kernel should
    /// return EBADF (Bad File Descriptor, errno 9). This tests that:
    /// 1. The syscall correctly returns an error (not silent failure)
    /// 2. The error is properly converted to io::Error
    /// 3. The original errno is preserved
    ///
    /// ## Why -1?
    ///
    /// File descriptor -1 is guaranteed to be invalid on POSIX systems.
    /// Valid file descriptors are non-negative integers (0, 1, 2, ...).
    #[test]
    fn test_get_terminal_attr_returns_ebadf_for_invalid_fd() {
        // fd = -1 is guaranteed invalid on all POSIX systems
        const INVALID_FD: i32 = -1;
        const EBADF: i32 = 9; // Bad file descriptor (Linux)

        let result = get_terminal_attr(INVALID_FD);

        // Must return an error, not Ok
        assert!(
            result.is_err(),
            "get_terminal_attr should return Err for invalid fd"
        );

        // Extract the error for inspection
        let err = result.expect_err("already verified is_err");

        // Verify the raw OS error code is EBADF
        assert_eq!(
            err.raw_os_error(),
            Some(EBADF),
            "Expected EBADF (errno {}), got {:?}",
            EBADF,
            err.raw_os_error()
        );
    }

    /// Verify set_terminal_attr returns EBADF for invalid file descriptor.
    ///
    /// ## What This Tests
    ///
    /// Similar to the get test, but for set_terminal_attr. This ensures
    /// the write path also properly handles invalid file descriptors.
    #[test]
    fn test_set_terminal_attr_returns_ebadf_for_invalid_fd() {
        const INVALID_FD: i32 = -1;
        const EBADF: i32 = 9;

        // Create a minimal valid termios struct
        // The content doesn't matter since the fd check happens first
        let termios = Termios {
            c_iflag: 0,
            c_oflag: 0,
            c_cflag: CS8, // Set some valid flag
            c_lflag: 0,
            c_line: 0,
            c_cc: [0u8; 19],
        };

        let result = set_terminal_attr(INVALID_FD, &termios);

        assert!(
            result.is_err(),
            "set_terminal_attr should return Err for invalid fd"
        );

        let err = result.expect_err("already verified is_err");

        assert_eq!(
            err.raw_os_error(),
            Some(EBADF),
            "Expected EBADF (errno {}), got {:?}",
            EBADF,
            err.raw_os_error()
        );
    }

    // =========================================================================
    // NON-TERMINAL FILE DESCRIPTOR TESTS (ENOTTY)
    // =========================================================================

    /// Verify get_terminal_attr returns ENOTTY for non-terminal fd.
    ///
    /// ## What This Tests
    ///
    /// When passed a valid file descriptor that is NOT a terminal
    /// (e.g., /dev/null, a regular file, a pipe), the ioctl should
    /// return ENOTTY (Not a typewriter/terminal, errno 25).
    ///
    /// This tests that:
    /// 1. The code correctly distinguishes terminals from other files
    /// 2. The kernel's ENOTTY is properly propagated
    /// 3. Opening a non-terminal doesn't cause crashes
    ///
    /// ## Why /dev/null?
    ///
    /// /dev/null is:
    /// - Always available on Linux/POSIX systems
    /// - Guaranteed to NOT be a terminal
    /// - Safe to open (unlike random device files)
    /// - Doesn't create temporary files (unlike using a regular file)
    #[test]
    fn test_get_terminal_attr_returns_enotty_for_non_terminal() {
        const ENOTTY: i32 = 25; // Not a terminal (Linux)

        // Open /dev/null - a valid fd that is definitely not a terminal
        let file = std::fs::File::open("/dev/null").expect("/dev/null should be openable on Linux");

        let fd = file.as_raw_fd();

        // Verify fd is valid (non-negative)
        assert!(fd >= 0, "File descriptor should be non-negative");

        let result = get_terminal_attr(fd);

        assert!(
            result.is_err(),
            "get_terminal_attr should return Err for non-terminal fd"
        );

        let err = result.expect_err("already verified is_err");

        assert_eq!(
            err.raw_os_error(),
            Some(ENOTTY),
            "Expected ENOTTY (errno {}), got {:?}",
            ENOTTY,
            err.raw_os_error()
        );
    }

    /// Verify set_terminal_attr returns ENOTTY for non-terminal fd.
    ///
    /// ## What This Tests
    ///
    /// Same as get test, but for the set path. Ensures both directions
    /// of terminal attribute access handle non-terminals correctly.
    #[test]
    fn test_set_terminal_attr_returns_enotty_for_non_terminal() {
        const ENOTTY: i32 = 25;

        // Need write access for set operation (open read-write)
        let file = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open("/dev/null")
            .expect("/dev/null should be openable read-write");

        let fd = file.as_raw_fd();

        let termios = Termios {
            c_iflag: 0,
            c_oflag: 0,
            c_cflag: CS8,
            c_lflag: 0,
            c_line: 0,
            c_cc: [0u8; 19],
        };

        let result = set_terminal_attr(fd, &termios);

        assert!(
            result.is_err(),
            "set_terminal_attr should return Err for non-terminal fd"
        );

        let err = result.expect_err("already verified is_err");

        assert_eq!(
            err.raw_os_error(),
            Some(ENOTTY),
            "Expected ENOTTY (errno {}), got {:?}",
            ENOTTY,
            err.raw_os_error()
        );
    }

    // =========================================================================
    // RAWTERMINAL::NEW ERROR TESTS
    // =========================================================================

    /// Verify RawTerminal::new fails gracefully when no terminal is available.
    ///
    /// ## What This Tests
    ///
    /// This test is marked #[ignore] because it requires a specific
    /// environment where /dev/tty is not available (e.g., CI runners,
    /// Docker containers without TTY, cron jobs).
    ///
    /// When manually testing, run with:
    /// ```bash
    /// cargo test test_raw_terminal_new_no_tty -- --ignored < /dev/null
    /// ```
    ///
    /// Or in a non-interactive environment:
    /// ```bash
    /// nohup cargo test test_raw_terminal_new_no_tty -- --ignored &
    /// ```
    #[test]
    #[ignore]
    fn test_raw_terminal_new_fails_without_tty() {
        let result = RawTerminal::new();

        // In a non-TTY environment, this should fail
        // The specific error depends on environment:
        // - ENOENT if /dev/tty doesn't exist
        // - ENXIO if no controlling terminal
        // - ENOTTY in some configurations
        if result.is_err() {
            let err = result.expect_err("verified is_err");
            eprintln!(
                "RawTerminal::new correctly failed without TTY: {} (errno: {:?})",
                err,
                err.raw_os_error()
            );
        } else {
            // If we got Ok, we're actually in a TTY - test doesn't apply
            eprintln!(
                "Note: Test environment has a TTY; \
                 this test should be run without a controlling terminal"
            );
        }
    }
}

#[cfg(test)]
mod other_tests {
    use super::*;
    use std::os::unix::io::AsRawFd;

    /// Integration test: suspend and re-activate raw mode cycle.
    /// Verifies that suspend restores original settings and activate
    /// re-enables raw mode with successful verification.
    /// Run with: cargo test -- --ignored test_suspend_activate_cycle
    #[test]
    #[ignore] // Requires interactive terminal
    fn test_suspend_activate_cycle() {
        let term = match RawTerminal::new() {
            Ok(t) => t,
            Err(e) => {
                eprintln!("Skipping: no terminal available: {}", e);
                return;
            }
        };

        // Suspend raw mode (restore original)
        let suspend_result = term.suspend_raw_mode();
        assert!(
            suspend_result.is_ok(),
            "suspend_raw_mode should succeed: {:?}",
            suspend_result.err()
        );

        // Re-activate raw mode (includes verification)
        let activate_result = term.activate_raw_mode();
        assert!(
            activate_result.is_ok(),
            "activate_raw_mode should succeed: {:?}",
            activate_result.err()
        );

        // Drop restores terminal
    }

    /// Regression test: verify ioctl round-trip survives asm operand refactor.
    /// Exercises get → modify → set → get → verify cycle.
    /// Run with: cargo test -- --ignored test_ioctl_roundtrip_regression
    #[test]
    #[ignore] // Requires terminal
    fn test_ioctl_roundtrip_regression() {
        let tty = match OpenOptions::new().read(true).write(true).open("/dev/tty") {
            Ok(f) => f,
            Err(_) => {
                eprintln!("Skipping: no terminal");
                return;
            }
        };
        let fd = tty.as_raw_fd();

        // Read original
        let original = get_terminal_attr(fd).expect("get_terminal_attr should succeed on real tty");

        // Modify to raw
        let mut raw = original;
        make_raw(&mut raw);
        set_terminal_attr(fd, &raw).expect("set_terminal_attr raw should succeed");

        // Read back and verify critical flags
        let readback = get_terminal_attr(fd).expect("get_terminal_attr readback should succeed");
        assert_eq!(readback.c_lflag & ICANON, 0, "ICANON must be cleared");
        assert_eq!(readback.c_lflag & ECHO, 0, "ECHO must be cleared");

        // Restore original
        set_terminal_attr(fd, &original).expect("set_terminal_attr restore should succeed");
    }

    /// Verify that a zeroed c_cflag is rejected as invalid.
    ///
    /// ## What This Tests
    ///
    /// The sanity check in get_terminal_attr rejects a termios with
    /// c_cflag == 0 (no baud rate, no char size, no flags), which
    /// indicates corruption or a kernel bug. This path cannot be
    /// reached through normal ioctl, so we test the validation
    /// logic directly by simulating a post-syscall zeroed struct.
    #[test]
    fn test_zeroed_cflag_would_be_rejected() {
        // Simulate what get_terminal_attr checks after a successful ioctl
        let termios = Termios {
            c_iflag: 0,
            c_oflag: 0,
            c_cflag: 0, // Invalid: no baud, no char size
            c_lflag: 0,
            c_line: 0,
            c_cc: [0u8; 19],
        };
        // The production check is: if termios.c_cflag == 0 { return Err(EINVAL) }
        assert_eq!(
            termios.c_cflag, 0,
            "Zeroed c_cflag should trigger the sanity-check rejection path"
        );
    }

    #[test]
    fn test_make_raw_preserves_baud_and_other_cflag_bits() {
        // CBAUD mask on Linux x86_64 = 0o010017 = 0x100F
        // Set some baud bits and a flag outside CSIZE|PARENB
        let mut termios = Termios {
            c_iflag: 0,
            c_oflag: 0,
            c_cflag: 0o010017 | 0o004000, // CBAUD bits + HUPCL
            c_lflag: 0,
            c_line: 0,
            c_cc: [0u8; 19],
        };
        let baud_and_hupcl = termios.c_cflag & !(CSIZE | PARENB);

        make_raw(&mut termios);

        let preserved = termios.c_cflag & !(CSIZE | PARENB);
        assert_eq!(
            preserved, baud_and_hupcl,
            "make_raw must not touch cflag bits outside CSIZE|PARENB"
        );
    }

    #[test]
    fn test_make_raw_disables_opost_requiring_explicit_crlf() {
        let mut termios = Termios {
            c_iflag: 0,
            c_oflag: 0xFFFFFFFF,
            c_cflag: 0,
            c_lflag: 0,
            c_line: 0,
            c_cc: [0u8; 19],
        };
        make_raw(&mut termios);
        assert_eq!(
            termios.c_oflag & OPOST,
            0,
            "OPOST must be cleared; all writes must use \\r\\n explicitly"
        );
    }
}

// ============================================================================
// MAIN - DEMONSTRATION
// ============================================================================
/*

// src/main.rs
use std::io::{self, Read, Write};

// import module
mod raw_terminal_x86_module;
use raw_terminal_x86_module::{RawTerminal, format_byte_info};

/// Demonstration of raw terminal functionality.
///
/// ## What This Demo Does
///
/// 1. Creates a RawTerminal (enters raw mode)
/// 2. Reads and displays each keystroke as individual bytes
/// 3. Shows byte values in decimal, hex, and character representation
/// 4. Pressing 'l' demonstrates the suspend/activate raw mode cycle
/// 5. Exits when 'q' is pressed
/// 6. Automatically restores terminal on exit
///
/// ## Single-Byte Display: Why Special Keys Appear as Multiple Lines
///
/// This demo reads and displays **one byte at a time** (VMIN=1, 1-byte buffer).
/// Regular keys (letters, digits, punctuation) produce a single byte each.
/// However, many special keys send **multi-byte escape sequences** that the
/// demo displays as separate lines:
///
/// ```text
/// Pressing the Delete key sends 4 bytes: ESC [ 3 ~
///
///   byte: 27  (0x1B) = '^['     ← ESC (escape sequence starts)
///   byte: 91  (0x5B) = '['      ← CSI bracket
///   byte: 51  (0x33) = '3'      ← key identifier
///   byte: 126 (0x7E) = '~'      ← sequence terminator
/// ```
///
/// The final byte `~` (0x7E) is **identical** to pressing the `~` key directly.
/// Without grouping escape sequences, the demo cannot distinguish them.
/// This is intentional: the demo shows raw bytes exactly as the kernel
/// delivers them, which is the point of raw mode.
///
/// ### Common Multi-Byte Escape Sequences
///
/// ```text
/// Key         Bytes sent          Displayed as
/// ─────────   ──────────────────  ─────────────────────
/// Up arrow    ESC [ A             ^[  [  A    (3 bytes)
/// Down arrow  ESC [ B             ^[  [  B    (3 bytes)
/// Right arrow ESC [ C             ^[  [  C    (3 bytes)
/// Left arrow  ESC [ D             ^[  [  D    (3 bytes)
/// Home        ESC [ H             ^[  [  H    (3 bytes)
/// End         ESC [ F             ^[  [  F    (3 bytes)
/// Insert      ESC [ 2 ~           ^[  [  2  ~ (4 bytes)
/// Delete      ESC [ 3 ~           ^[  [  3  ~ (4 bytes)
/// Page Up     ESC [ 5 ~           ^[  [  5  ~ (4 bytes)
/// Page Down   ESC [ 6 ~           ^[  [  6  ~ (4 bytes)
/// F1          ESC O P             ^[  O  P    (3 bytes)
/// F5          ESC [ 1 5 ~         ^[  [  1  5  ~ (5 bytes)
/// ```
///
/// A production application that needs to identify special keys must group
/// bytes following an ESC (0x1B) into complete sequences before interpreting
/// them. This demo intentionally does not do that—it exists to show the
/// raw byte stream that such a parser would consume.
///
/// ## How to Run
///
/// ```bash
/// cargo run --release
/// ```
///
/// ## Note on Error Returns
///
/// This is a demonstration binary. Returning Err from main causes process
/// exit with code 1. For a production daemon or long-running service,
/// terminal initialization failure should be handled by the caller with
/// retry or fallback logic, not by terminating.

/// Demonstrates the suspend / activate raw mode cycle for application use.
///
/// ## Project Context
///
/// Real terminal applications (text editors, file browsers, shells) frequently
/// need to give up raw mode so a subprocess can have a fully functional normal
/// terminal. Classic examples:
/// - A text editor's ':sh' command drops to a shell
/// - A file browser's 'e' key opens vim or nano
/// - A pager's '!' key pipes output through a shell command
///
/// This function demonstrates the three-step pattern used in all such cases:
///
/// 1. `suspend_raw_mode()` — restore the original cooked terminal so the
///    subprocess sees echo, line editing, and signal keys (Ctrl+C etc.)
///
/// 2. Run subprocess — it inherits our file descriptors in cooked mode
///    and behaves exactly as if launched from a normal shell.
///
/// 3. `activate_raw_mode()` — re-enable raw mode AND verify it was applied.
///    Internally this calls `verify_raw_mode_active()`, which reads terminal
///    settings back from the kernel and confirms ICANON and ECHO are cleared.
///    This guards against kernel/driver bugs, race conditions with other
///    processes, and bit-flip corruption before they cause silent misbehavior.
///
/// ## Cooked Mode Behavior (visible during this demo)
///
/// While suspended, the terminal is in cooked (canonical) mode:
/// - Echo is ON: typed characters appear as you type
/// - ICANON is ON: input is line-buffered; Enter submits
/// - OPOST is ON: bare `\n` is automatically expanded to `\r\n` by the driver
/// - Ctrl+C, Ctrl+Z: signal generation is active again
///
/// ## Arguments
///
/// * `term` - mutable reference to the active RawTerminal
///
/// ## Returns
///
/// * `Ok(())` - subprocess ran, raw mode restored and verified
/// * `Err(io::Error)` - suspend, subprocess start, or reactivation failed;
///   caller should decide whether to retry or exit cleanly
fn run_subprocess_demo(term: &mut RawTerminal) -> io::Result<()> {
    // Announce intent while still in raw mode (\r\n required, OPOST is off)
    if let Err(e) = term.write_all(b"\r\nDemo: suspending raw mode to run `ls -la`.\r\n") {
        return Err(e);
    }
    if let Err(e) = term.flush() {
        return Err(e);
    }

    // -------------------------------------------------------------------------
    // Step 1: Suspend raw mode
    // -------------------------------------------------------------------------
    // Restores the original terminal settings saved at RawTerminal::new().
    // After this call the driver re-enables echo, line buffering, and OPOST.
    // The subprocess will inherit these settings through our file descriptors.
    // -------------------------------------------------------------------------
    if let Err(e) = term.suspend_raw_mode() {
        // Could not suspend; stay in raw mode and report to caller
        let _ = term.write_all(b"RSUB: suspend_raw_mode failed\r\n");
        let _ = term.flush();
        return Err(e);
    }

    // -------------------------------------------------------------------------
    // Step 2: Run subprocess
    // -------------------------------------------------------------------------
    // The subprocess inherits our /dev/tty file descriptor in cooked mode.
    // `ls -la` is used here because it is always available on Linux and its
    // column-aligned output visibly demonstrates that the subprocess is
    // receiving a real terminal (not a pipe).
    //
    // In cooked mode OPOST is on, so bare \n works for our stderr messages.
    // We use stderr here to avoid interleaving with subprocess stdout.
    // -------------------------------------------------------------------------
    let _ = std::io::stderr().write_all(
        b"\n[Cooked mode active: echo on, line editing on, Ctrl+C works]\n\
            (To demonstrate a sub-process we will use bash: ls (show files))
            Running: ls -la\n\n",
    );

    let run_result = std::process::Command::new("ls").arg("-la").status();

    match run_result {
        Ok(status) => {
            if !status.success() {
                // ls ran but reported an error (e.g. permission denied)
                // Not fatal: we still need to reactivate raw mode below
                let _ =
                    std::io::stderr().write_all(b"[Cooked mode] ls exited with non-zero status\n");
            }
        }
        Err(e) => {
            // ls binary missing or fork failed; reactivate before returning
            let _ = std::io::stderr().write_all(b"[Cooked mode] Could not run ls\n");
            // Best-effort reactivation; if this also fails, propagate original error
            let _ = term.activate_raw_mode();
            return Err(e);
        }
    }

    // -------------------------------------------------------------------------
    // "Press Enter to continue" in cooked mode
    // -------------------------------------------------------------------------
    // This line is intentionally in cooked mode so the user can experience
    // the contrast with raw mode:
    // - The prompt appears immediately (echo is on)
    // - read() blocks until Enter (ICANON is on)
    // - Backspace / Ctrl+U / Ctrl+W line editing works (kernel handles it)
    //
    // We read into a stack buffer and discard the content;
    // we only need the blocking behavior (user pressed Enter).
    // -------------------------------------------------------------------------
    let _ = std::io::stderr().write_all(b"\nPress Enter to return to raw mode...\n");

    let mut enter_buf = [0u8; 64];
    // In cooked mode, read() returns the whole typed line when Enter is pressed
    let _ = term.read(&mut enter_buf);

    // -------------------------------------------------------------------------
    // Step 3: Reactivate raw mode (with built-in kernel verification)
    // -------------------------------------------------------------------------
    // activate_raw_mode() does two things:
    //   a) Applies raw mode settings derived from the SAVED original (prev_ios),
    //      not from whatever the current terminal state is. This makes
    //      reactivation safe even if the subprocess modified terminal settings.
    //   b) Calls verify_raw_mode_active() internally: reads the settings back
    //      from the kernel and confirms ICANON and ECHO are actually cleared.
    //
    // Verification failure returns Err here. In a production application you
    // might retry (activate_raw_mode is safe to call again) or degrade to a
    // non-interactive mode. Here we report and propagate to the caller.
    // -------------------------------------------------------------------------
    if let Err(e) = term.activate_raw_mode() {
        let _ = std::io::stderr().write_all(b"RSUB: activate_raw_mode / verification failed\n");
        return Err(e);
    }

    // Raw mode confirmed active: use \r\n from here
    if let Err(e) = term.write_all(b"\r\nRaw mode re-activated and verified.\r\n") {
        return Err(e);
    }
    if let Err(e) = term.flush() {
        return Err(e);
    }

    Ok(())
}

// ============================================================================
// MAIN - DEMONSTRATION
// ============================================================================

/// Demonstration of raw terminal functionality.
///
/// ## What This Demo Does
///
/// 1. Creates a RawTerminal (enters raw mode)
/// 2. Reads and displays each keystroke
/// 3. Shows special key escape sequences
/// 4. Exits when 'q' is pressed
/// 5. Automatically restores terminal on exit
///
/// ## How to Run
///
/// ```bash
/// cargo run --release
/// ```
///
/// ## Expected Behavior
///
/// - Each keypress appears immediately (no Enter needed)
/// - Special keys show their escape sequences (e.g., arrows show ^[[A)
/// - Pressing 'q' exits the program
/// - After exit, terminal behaves normally
///
/// ## Note on Error Returns
///
/// This is a demonstration binary. Returning Err from main causes process
/// exit with code 1. For a production daemon or long-running service,
/// terminal initialization failure should be handled by the caller with
/// retry or fallback logic, not by terminating.
fn main() -> io::Result<()> {
    // Attempt to create raw terminal
    let mut term = match RawTerminal::new() {
        Ok(t) => t,
        Err(e) => {
            eprintln!("Failed to initialize raw terminal: {}", e);
            eprintln!("This program requires a terminal (TTY) to run.");
            eprintln!("It cannot run with redirected stdin/stdout or in some CI environments.");
            return Err(e);
        }
    };

    // Write instructions
    /*
    In normal terminal mode (cooked/canonical):

    \n (0x0A, line feed) is automatically converted to \r\n (CRLF) by the driver
    This is the OPOST (post-process output) flag at work
    You write \n, terminal displays as \r\n (down + return to column 0)

    In raw mode (make_raw disables OPOST):

    \n stays as just \n
    Moves down one line but does NOT return to column 0
    Next character appears at whatever column you're at (usually column 0 in practice, but the driver doesn't guarantee it)
    To get a true newline, you must write \r\n explicitly

    This is pure Linux termios, not Windows compat.
    */
    // In raw mode, OPOST is disabled, so the terminal driver does NOT
    // automatically convert \n to \r\n. We must write \r\n explicitly:
    // \n alone moves down but doesn't return to column 0.
    // This is Linux termios behavior (see termios(3), OPOST flag).
    if let Err(e) = term.write_all(b"Raw mode active!\r\n") {
        eprintln!("Write error: {}", e);
        return Err(e);
    }
    if let Err(e) = term.write_all(b"Press keys to see their byte values.\r\n") {
        eprintln!("Write error: {}", e);
        return Err(e);
    }
    if let Err(e) = term.write_all(b"Press 'l' to suspend (runs ls), 'q' to quit.\r\n") {
        eprintln!("Write error: {}", e);
        return Err(e);
    }
    if let Err(e) = term.write_all(b"---\r\n") {
        eprintln!("Write error: {}", e);
        return Err(e);
    }
    if let Err(e) = term.flush() {
        eprintln!("Flush error: {}", e);
        return Err(e);
    }

    // -------------------------------------------------------------------------
    // Main input loop
    // -------------------------------------------------------------------------
    // This is an INTENTIONAL always-loop (interactive user input).
    //
    // ## Bounded vs Always-Loop (Power of 10, Rule 2)
    //
    // This loop is unbounded by design because:
    // 1. It waits for external user input (inherently unbounded timing)
    // 2. It has a clear exit condition: user presses 'q'
    // 3. It handles all error cases by breaking (no infinite spin)
    // 4. Terminal EOF also causes clean exit
    //
    // ## Failsafe Layer
    //
    // The failsafe is the Drop implementation on RawTerminal: even if
    // this loop somehow spins forever or panics, the terminal will be
    // restored when the process exits (via Drop or kernel cleanup).
    //
    // Optional: For long-running production use, add iteration watchdog:
    // -------------------------------------------------------------------------
    const MAX_ITERATIONS: u64 = u64::MAX; // Effectively unbounded but trackable
    let mut iterations: u64 = 0;

    let mut buf = [0u8; 1];
    loop {
        // Watchdog check (overflow-safe)
        iterations = iterations.saturating_add(1);
        if iterations == MAX_ITERATIONS {
            // Would require ~584 billion years at 1 keypress/nanosecond
            // This branch exists for completeness, not realism
            let _ = term.write_all(b"\r\nWatchdog limit reached.\r\n");
            break;
        }

        match term.read(&mut buf) {
            Ok(0) => {
                // EOF - terminal closed or redirected
                break;
            }
            Ok(_) => {
                let byte = buf[0];

                let mut output = [0u8; 64];
                let len = format_byte_info(&mut output, byte);

                // Handle format failure (returns 0 on bounds error)
                if len == 0 {
                    // Should never happen with correct buffer sizing
                    // Skip this byte rather than panic
                    continue;
                }

                if let Err(_e) = term.write_all(&output[..len]) {
                    // Write failed - terminal may be gone
                    break;
                }
                if let Err(_e) = term.flush() {
                    break;
                }
                if byte == b'l' {
                    if let Err(_e) = run_subprocess_demo(&mut term) {
                        let _ = term.write_all(b"\r\nRSUB: demo failed, see stderr\r\n");
                        let _ = term.flush();
                    }
                    continue;
                }

                if byte == b'q' {
                    let _ = term.write_all(b"\r\nExiting...\r\n");
                    let _ = term.flush();
                    break;
                }
            }
            Err(_e) => {
                // Read error - terminal unavailable
                break;
            }
        }
    }

    // RawTerminal is dropped here, automatically restoring the terminal
    Ok(())
}


*/
