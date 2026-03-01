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
