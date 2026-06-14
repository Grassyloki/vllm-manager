"""
tui_lib.py — shared curses helpers for vllm_manager.py and manage_models.py.

Two managers share the same colour scheme, list/checkbox selector, text input
prompt, and 'leave curses to run a CLI command then come back' wrapper. This
module owns those primitives so a fix lands once.

Usage:
    import tui_lib as tui

    def _ui(stdscr):
        tui.init_colors()
        idx = tui.select(stdscr, "Pick one", [("a", 0), ("b", 0)])
        ...

    tui.launch(_ui)
"""
from __future__ import annotations
import os
import sys


# == Colour pairs ============================================================
# Initialised by init_colors() inside a curses context. Use via curses.color_pair(C_*).

C_TITLE  = 1
C_GREEN  = 2
C_YELLOW = 3
C_DIM    = 4
C_CYAN   = 5
C_RED    = 6


def init_colors():
    import curses
    curses.start_color()
    curses.use_default_colors()
    curses.init_pair(C_TITLE,  curses.COLOR_CYAN,   -1)
    curses.init_pair(C_GREEN,  curses.COLOR_GREEN,  -1)
    curses.init_pair(C_YELLOW, curses.COLOR_YELLOW, -1)
    curses.init_pair(C_DIM,    curses.COLOR_WHITE,  -1)
    curses.init_pair(C_CYAN,   curses.COLOR_CYAN,   -1)
    curses.init_pair(C_RED,    curses.COLOR_RED,    -1)


# == Drawing primitives ======================================================

def addstr(win, y, x, text, attr=0):
    """Write text, silently truncating at the screen edge."""
    h, w = win.getmaxyx()
    if y < 0 or y >= h or x >= w:
        return
    try:
        win.addnstr(y, x, text, max(0, w - x - 1), attr)
    except Exception:
        pass


# == Selector widget =========================================================

def select(stdscr, title, items, *, header=None, multi=False,
           refresh_cb=None, refresh_ms=5000):
    """
    Interactive list / checkbox selector.

    items:       list[(label, attr)] — attr is a curses attribute or 0.
    header:      optional list[(line, attr)] drawn above the items.
    multi=False: Enter picks one, returns index or -1 on cancel.
    multi=True:  Space toggles, Enter confirms, returns list[int] or [] on cancel.
    refresh_cb:  optional callable returning (new_header, new_items) every
                 refresh_ms ms while idle. Either may be None to leave unchanged.

    Keys: Up/Down/Home/End to move, Esc or q to cancel.
    """
    import curses

    if not items:
        return [] if multi else -1

    sel = 0
    checks = [False] * len(items) if multi else None

    if refresh_cb is not None:
        stdscr.timeout(refresh_ms)
    else:
        stdscr.timeout(-1)

    while True:
        stdscr.erase()
        h, w = stdscr.getmaxyx()

        addstr(stdscr, 1, 2, title,
               curses.A_BOLD | curses.color_pair(C_TITLE))
        y = 3

        if header:
            for entry in header:
                if y >= h - 3:
                    break
                if isinstance(entry, list):
                    # segmented line: [(text, attr), ...] drawn left-to-right,
                    # so one line can mix colours (e.g. a stacked usage bar).
                    x = 2
                    for seg_text, seg_attr in entry:
                        addstr(stdscr, y, x, seg_text, seg_attr)
                        x += len(seg_text)
                else:
                    hline, hattr = entry
                    addstr(stdscr, y, 2, hline, hattr)
                y += 1
            y += 1
            addstr(stdscr, y, 2, "─" * min(w - 4, 56), curses.A_DIM)
            y += 2

        item_y_start = y
        item_area = max(1, h - item_y_start - 2)

        if len(items) <= item_area:
            offset = 0
        elif sel < item_area // 2:
            offset = 0
        elif sel >= len(items) - (item_area - item_area // 2):
            offset = max(0, len(items) - item_area)
        else:
            offset = sel - item_area // 2

        end = min(offset + item_area, len(items))

        for i in range(offset, end):
            if y >= h - 2:
                break

            label, item_attr = items[i]

            if checks is not None:
                mark = "x" if checks[i] else " "
                prefix = f"[{mark}] "
            else:
                prefix = ""

            pointer = "> " if i == sel else "  "
            line = f" {pointer}{prefix}{label}"

            if i == sel:
                padded = line.ljust(min(len(line) + 2, w - 2))
                addstr(stdscr, y, 1, padded,
                       curses.A_REVERSE | curses.A_BOLD)
            else:
                addstr(stdscr, y, 1, line, item_attr)
            y += 1

        if offset > 0:
            addstr(stdscr, item_y_start, w - 4, " ↑ ", curses.A_DIM)
        if end < len(items):
            addstr(stdscr, min(y - 1, h - 3), w - 4, " ↓ ", curses.A_DIM)

        if multi:
            foot = (" [↑↓] Navigate  [Space] Toggle  "
                    "[Enter] Confirm  [Esc] Back")
        else:
            foot = " [↑↓] Navigate  [Enter] Select  [Esc] Back"
        addstr(stdscr, h - 1, 0, foot, curses.A_DIM)

        stdscr.refresh()

        key = stdscr.getch()

        if key == curses.KEY_UP:
            sel = max(0, sel - 1)
        elif key == curses.KEY_DOWN:
            sel = min(len(items) - 1, sel + 1)
        elif key == ord(' ') and multi:
            checks[sel] = not checks[sel]
        elif key in (curses.KEY_ENTER, 10, 13):
            stdscr.timeout(-1)
            if multi:
                return [i for i, c in enumerate(checks) if c]
            return sel
        elif key in (27, ord('q')):
            stdscr.timeout(-1)
            return [] if multi else -1
        elif key == curses.KEY_HOME:
            sel = 0
        elif key == curses.KEY_END:
            sel = len(items) - 1
        elif key == curses.KEY_RESIZE:
            pass  # redraw on next iteration
        elif key == -1 and refresh_cb is not None:
            try:
                new_header, new_items = refresh_cb()
            except Exception:
                new_header, new_items = None, None
            if new_items is not None:
                items = new_items
                if multi:
                    if len(checks) < len(items):
                        checks = checks + [False] * (len(items) - len(checks))
                    elif len(checks) > len(items):
                        checks = checks[:len(items)]
                sel = min(sel, max(0, len(items) - 1))
            if new_header is not None:
                header = new_header


# == Yes/No prompt ===========================================================

def confirm(stdscr, question, *, default=True, explain=None):
    """Inline Y/n prompt. `explain` is an optional list of context lines."""
    items = [
        ("Yes", 0),
        ("No",  0),
    ]
    header = None
    if explain:
        header = [(line, 0) for line in explain]
    title = question + ("  [default: Yes]" if default else "  [default: No]")
    idx = select(stdscr, title, items, header=header)
    if idx == -1:
        return default
    return idx == 0


# == Text input ==============================================================

def text(stdscr, prompt, *, width=None, default=""):
    """One-line text prompt at the bottom of the screen.

    `prompt` is rendered verbatim — caller controls spacing/colon. Empty
    input returns `default` (which is "" by default).
    """
    import curses
    curses.echo()
    curses.curs_set(1)
    h, w = stdscr.getmaxyx()
    y = h - 2
    label = prompt
    max_w = width or max(8, w - len(label) - 4)
    addstr(stdscr, y, 2, label, curses.A_BOLD)
    stdscr.refresh()
    try:
        raw = stdscr.getstr(y, 2 + len(label), max_w)
        s = raw.decode("utf-8").strip() if raw else ""
        return s or default
    except Exception:
        return default
    finally:
        curses.noecho()
        curses.curs_set(0)
        stdscr.touchwin()
        stdscr.refresh()


# == Pause / shell-out =======================================================

def pause(stdscr, msg="Press Enter to continue ..."):
    import curses
    curses.endwin()
    try:
        input(f"\n{msg}")
    except (EOFError, KeyboardInterrupt):
        print()
    stdscr.touchwin()
    stdscr.refresh()


def run_cmd(stdscr, callback, *args, pause_after=True):
    """Leave curses, run callback (which prints to stdout), restore curses."""
    import curses
    curses.endwin()
    print()
    try:
        callback(*args)
    except KeyboardInterrupt:
        print("\n  Interrupted.")
    except SystemExit as e:
        # Commands call sys.exit() on failure; don't let it unwind through
        # curses.wrapper (which then crashes in endwin()). Report and return.
        if e.code not in (0, None):
            print(f"\n  (command exited with status {e.code})")
    if pause_after:
        try:
            input("\nPress Enter to continue ...")
        except (EOFError, KeyboardInterrupt):
            print()
    stdscr.touchwin()
    stdscr.refresh()


def shell_out(stdscr, argv, *, pause_after=True):
    """Leave curses, execvp-style shell-out, then come back."""
    import curses
    import subprocess
    curses.endwin()
    print()
    try:
        subprocess.run(argv)
    except KeyboardInterrupt:
        print("\n  Interrupted.")
    if pause_after:
        try:
            input("\nPress Enter to continue ...")
        except (EOFError, KeyboardInterrupt):
            print()
    stdscr.touchwin()
    stdscr.refresh()


# == Top-level launcher ======================================================

def launch(main_fn):
    """Wrap curses.wrapper with a graceful 'curses missing' message."""
    try:
        import curses
    except ImportError:
        print("curses module not available. Use CLI commands instead.")
        sys.exit(1)
    curses.wrapper(main_fn)


# == Terminal-size guard =====================================================

def too_small(stdscr, min_h=10, min_w=50):
    h, w = stdscr.getmaxyx()
    return h < min_h or w < min_w
