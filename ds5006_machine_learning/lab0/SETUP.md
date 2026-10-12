# DS5006 — Setting up your environment

**Use VS Code.** That is what will be provided during the two lab exams, so the time you
spend getting comfortable with it now is time you do not spend fighting an unfamiliar
editor under exam conditions. Everything below assumes it.

This takes about ten minutes and you do it once for the semester. Do it in week 1, not
the night Lab 0 is due — if something is wrong with your machine, finding out now is
most of the point of Lab 0.

If you get stuck on any of this, email the instructor or a TA. Environment problems are
the least interesting way to lose marks and we would much rather sort them out early.

---

## 1. Install

**Python 3.11 or newer.** Check with `python3 --version` in a terminal. If it is missing
or older, install from [python.org](https://www.python.org/downloads/) or your package
manager. On Windows, tick **"Add Python to PATH"** during installation — skipping it
causes most of the problems people hit later.

**VS Code**, from [code.visualstudio.com](https://code.visualstudio.com/).

**The Python extension.** Open VS Code, click the Extensions icon in the left bar
(or `Ctrl+Shift+X`), search for "Python", and install the one published by Microsoft.

## 2. Make a folder and open it

Create a folder for the course, say `ds5006`, and put `lab0.py`, `test_public.py` and
`requirements.txt` directly inside it.

In VS Code: **File → Open Folder**, and choose `ds5006`.

> Open the *folder*, not the individual file. VS Code runs your code from the folder you
> have open, and `test_public.py` can only find `lab0.py` if both sit at the top level of
> it. This is the single most common setup mistake.

## 3. Create the environment

Open the Command Palette with `Ctrl+Shift+P` (`Cmd+Shift+P` on macOS), type
**Python: Create Environment**, and press Enter. You will be asked:

1. **Venv or Conda** — choose **Venv**.
2. **Which interpreter** — choose your Python 3.11+.
3. **Dependencies** — tick `requirements.txt` **if you are offered it**. You may not be.
   That is fine; step 4 installs them either way.

A `.venv` folder will then appear in your workspace.

### If you cannot find "Python: Create Environment"

It is provided by the Python extension, so if it is missing, work through these in
order. Nearly always it is one of the first three.

- **Is the extension installed and enabled?** `Ctrl+Shift+X`, then type `@installed
  python`. You want **Python**, published by Microsoft.
- **Is a folder open?** The command needs one. If you used File → *Open File* instead
  of File → *Open Folder*, it will not appear.
- **Is the window in Restricted Mode?** VS Code disables extensions in folders it does
  not trust, and every Python command disappears. Look for "Restricted Mode" in the
  status bar or a yellow banner, and trust the folder.
- **Right palette?** `Ctrl+Shift+P` is the command palette. `Ctrl+P` is file search —
  type `>` first if you are in that one.
- **Reload.** `Ctrl+Shift+P` → **Developer: Reload Window**.

If none of that works, the terminal route below does the same job in thirty seconds:

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
source .venv/bin/activate       # macOS or Linux
```

Then carry on from step 4. Do still fix the extension afterwards — without it you lose
autocomplete and the debugger, which are exactly what you will want in the lab exam.

## 4. Install NumPy and check it took

Look at the **bottom-right status bar**: it should show your Python version with
`.venv` beside it. If it does not, open the Command Palette and run
**Python: Select Interpreter**, then pick the one marked `.venv`.

Now open a terminal inside VS Code with ``Ctrl+` `` (the backtick key, top-left of most
keyboards) or **Terminal → New Terminal**. The prompt should begin with `(.venv)`.

> If it does not show `(.venv)`, close that terminal and open a new one. VS Code only
> activates the environment in terminals opened *after* the interpreter was selected.

Now install the packages. This is harmless if VS Code already installed them:

```bash
pip install -r requirements.txt
```

If that complains it cannot find the file, `requirements.txt` is not in the folder you
opened. Either move it there, or just run `pip install numpy matplotlib`.

> **On Windows**, File Explorer hides known file extensions, so a file saved from
> Notepad as `requirements.txt` is often really `requirements.txt.txt`. It looks
> correct in the file listing and nothing will find it. Turn on
> View → Show → File name extensions to check.

Finally:

```bash
python -c "import numpy; print(numpy.__version__)"
```

Anything starting with `2.` means you are set up correctly.

## 5. Check the lab runs

In that same terminal:

```bash
python test_public.py
```

On the untouched stub this reports ten unimplemented functions, flags `student_info` as
blank, and produces **no errors or tracebacks**. If you see an `ImportError`, a
`SyntaxError`, or a complaint about NumPy, the problem is your setup rather than your
code. Fix it before writing anything.

You can also press the **▶ Run** button at the top right of the editor, but the terminal
is more reliable and is what you will use in the lab exam.

---

## Worth learning before the exam

Ninety minutes is not long, and these three things save real time:

**The integrated terminal** (``Ctrl+` ``). Run your file, read the traceback, fix, repeat.
Faster than clicking anything.

**The debugger.** Click in the margin to the left of a line number to set a red
breakpoint, then press `F5`. Execution stops there and the **Variables** pane on the left
shows every array in scope with its shape. In a course where half of all bugs are an
array with the wrong shape, this beats scattering `print` statements — and it is much
faster under time pressure.

**The Command Palette** (`Ctrl+Shift+P`). Everything VS Code can do is in there. If you
remember nothing else, remember this shortcut.

---

## When you submit

Zip **only** `lab0.py` and `reflection.md`. Do **not** include the `.venv` folder — it is
hundreds of megabytes and it is not your work.

## Memory, and one thing that will confuse you

One Lab 0 task computes a 2000 × 2000 distance matrix. The answer is only about 32 MB,
but the most obvious way of computing it builds a 320 MB temporary along the way, so
your machine needs roughly 400 MB free to hold it comfortably.

If that task runs very slowly on an older or heavily loaded laptop, the cause may be
that your machine is swapping to disk rather than that your code is wrong. There is a
way of computing the same thing that never allocates the large temporary — finding it is
part of the exercise.

## If you would rather use Colab or Kaggle

They work, and NumPy is already installed. But you will be given VS Code in the lab
exams, so treat them as a fallback rather than a habit. Two things to watch:

- **Submit a `.py` file, not a notebook.** Export your work as `lab0.py`, with exactly
  that name. A notebook will not be marked.
- **Your file must import cleanly.** Nothing may run when it is imported: no stray
  `print` at the top level, no plotting, no reading files. Anything used to produce
  figures goes inside `if __name__ == "__main__":` at the bottom.

Your machine's speed does not matter. Timing is measured when your work is marked, on a
single machine, so everyone is compared on equal terms.

## Rules that follow from all this

- **NumPy and the standard library only** in the graded functions.
- **No network access** while your code runs.
- **No reading from disk.** Everything your functions need arrives as arguments.
- **Import must be silent.** No output, no side effects.

These are not arbitrary. Your file is imported and run automatically, and anything that
prints, downloads or writes during import will fail before it is ever marked.
