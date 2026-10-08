import io
import json
import re
from datetime import datetime, timezone
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox, blended_transform_factory
from matplotlib.ticker import MultipleLocator
import streamlit as st


# ---------------------------------------------------------------------------
# Data entry parsing
# ---------------------------------------------------------------------------

# A token that means "a session happened, but there is no data point."
# The hyphen is the taught marker. The dash and minus look-alikes cover text
# pasted from Word, and x / na are kept from earlier versions of the app.
DEFAULT_Y_LABEL = "% correct responses"

NO_DATA_TOKENS = {"-", "–", "—", "−", "x", "na", "n/a"}

DATA_BOX_HELP = (
    "Type one value per session, in order, separated by commas, spaces, or both. "
    "For a session that happened but has no data point, type a hyphen (-) in its place, "
    "as in 4, -, 5, or leave the spot between two commas empty, as in 4,,5. "
    "That session keeps its place on the x-axis and no point is drawn for it."
)


def parse_series_text(s):
    """Return (values, bad_tokens).

    Commas and whitespace both separate values. A lone hyphen, or an empty
    slot between two commas, is a session with no data (NaN). A hyphen joined
    to a number, as in -3, is a negative number. A stray comma at the very
    start or end of the entry is ignored rather than counted as a session.
    """
    if s is None or not s.strip():
        return [], []
    pieces = s.split(",")
    while pieces and not pieces[0].strip():
        pieces.pop(0)
    while pieces and not pieces[-1].strip():
        pieces.pop()

    values, bad = [], []
    for piece in pieces:
        tokens = piece.split()
        if not tokens:
            values.append(np.nan)
            continue
        for t in tokens:
            tt = t.strip().lower()
            if tt in NO_DATA_TOKENS:
                values.append(np.nan)
                continue
            try:
                values.append(float(tt.replace("−", "-")))
            except ValueError:
                bad.append(t)
    return values, bad


def parse_series(s):
    values, bad = parse_series_text(s)
    if bad:
        shown = ", ".join(f"'{b}'" for b in bad)
        st.error(
            f"Could not read {shown}. Use numbers, and a hyphen (-) for a session with no data point."
        )
        return []
    return values


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def distinct_measure_names(phase_measures):
    """Measure names in order of first appearance across phases."""
    names = []
    for measures in phase_measures:
        for mname, _ in measures:
            if mname not in names:
                names.append(mname)
    return names


def build_figure(
    phase_titles,
    phase_measures,
    graph_title="Single-Case Design Graph",
    y_label="% correct responses",
    x_label="Sessions",
    y_min=0.0,
    y_max=100.0,
    y_tick=10.0,
    x_tick=1.0,
    use_max_x=False,
    fixed_max_x=None,
    show_title=True,
    show_phase_titles=True,
    show_x=True,
    show_legend=True,
    connect_gaps=False,
    color_mode="Color",
    custom_colors=None,
    is_multiple_baseline=False,
    extend_phase_lines=False,
    stair_step_length=0.0,
):
    fig, ax = plt.subplots(figsize=(10, 5))
    # Fixed margins so plotting area width remains consistent across figures
    fig.set_constrained_layout(False)
    fig.set_tight_layout(False)
    fig.subplots_adjust(left=0.1, right=0.78, bottom=0.32, top=0.88)

    # The first six colors and markers are the ones the app has always used.
    # The rest give a phase with many measures a distinct color and marker.
    default_colors = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b",
        "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    ]
    grayscale_colors = ["#000000", "#555555", "#888888", "#AAAAAA", "#CCCCCC", "#EEEEEE"]
    markers = ["o", "s", "D", "^", "v", "P", "X", "*", "h", "<", ">", "p"]

    if color_mode == "Grayscale":
        palette = grayscale_colors
    else:
        palette = default_colors

    # Precompute phase lengths and starts. A session with no data still counts.
    phase_lengths = []
    for measures in phase_measures:
        max_len = max((len(m[1]) for m in measures if m[1]), default=0)
        phase_lengths.append(max_len)

    phase_starts = []
    cx = 1
    for L in phase_lengths:
        phase_starts.append(cx)
        cx += L

    y_top = y_max + (y_max - y_min) * 0.05
    plotted_xmax = 0.5
    phase_lines = []

    # One color, one marker, and one legend entry per distinct measure name.
    # A measure keeps the same color and marker in every phase.
    style_order = distinct_measure_names(phase_measures)
    labeled = set()
    for idx, (ptitle, measures) in enumerate(zip(phase_titles, phase_measures)):
        start_x = phase_starts[idx]
        L = phase_lengths[idx]

        for j, (mname, data) in enumerate(measures):
            if not data:
                continue
            x_vals = np.arange(start_x, start_x + len(data))
            y_vals = np.asarray(data, dtype=float)
            if connect_gaps:
                # Alternating treatments: join this measure's points across
                # the sessions that have no data point.
                keep = ~np.isnan(y_vals)
                px, py = x_vals[keep], y_vals[keep]
            else:
                # NaN breaks the line, so the data path stops at a gap.
                px, py = x_vals, y_vals
            k = style_order.index(mname)
            if color_mode == "Custom" and isinstance(custom_colors, dict) and mname in custom_colors:
                color = custom_colors[mname]
            else:
                color = palette[k % len(palette)]
            if mname in labeled:
                label = "_nolegend_"
            else:
                label = mname
                labeled.add(mname)
            ax.plot(
                px, py,
                color=color,
                marker=markers[k % len(markers)],
                label=label,
                linewidth=2,
                markersize=6,
            )
            if len(x_vals) > 0:
                plotted_xmax = max(plotted_xmax, x_vals[-1] + 0.5)

        if show_phase_titles and L > 0:
            ax.text(
                start_x, y_top, ptitle,
                ha="left", va="bottom",
                fontsize=10, fontweight="bold", wrap=True
            )

        if idx < len(phase_titles) - 1 and L > 0:
            # The phase change line sits halfway between the last session of
            # this phase and the first session of the next one.
            x_line = start_x + L - 0.5
            if is_multiple_baseline:
                # Drawn after the x range is set, so the stair-step can be
                # joined to the graphs above and below it.
                phase_lines.append(x_line)
            else:
                ax.axvline(x=x_line, color="black", linestyle="--", linewidth=1.5, zorder=3)

    if show_title:
        ax.set_title(graph_title, fontsize=13, pad=20)
    ax.set_ylabel(y_label)

    ax.set_ylim(y_min, y_max + (y_max - y_min) * 0.1)
    ax.set_yticks(np.arange(y_min, y_max + 1e-9, y_tick))
    ax.xaxis.set_major_locator(MultipleLocator(base=x_tick))

    # The x-axis line, tick marks, tick labels, and axis label move together.
    if show_x:
        ax.set_xlabel(x_label)
        ax.tick_params(axis="x", which="both", bottom=True, top=False, labelbottom=True)
    else:
        ax.tick_params(axis="x", which="both", bottom=False, top=False, labelbottom=False)
        ax.spines["bottom"].set_visible(False)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if use_max_x and fixed_max_x is not None:
        ax.set_xlim(0.5, max(fixed_max_x + 0.5, plotted_xmax))
    else:
        ax.set_xlim(0.5, max(plotted_xmax, 2.5))

    handles, labels = ax.get_legend_handles_labels()
    if show_legend and labels:
        ax.legend(frameon=False, loc="center left", bbox_to_anchor=(1.02, 0.5))

    if is_multiple_baseline:
        _finish_multiple_baseline(
            fig, ax, phase_lines, show_title, show_x,
            extend_phase_lines and stair_step_length > 0, stair_step_length,
        )

    return fig, ax, handles, labels


# Line width of the phase change lines, in points.
PHASE_LINE_WIDTH = 1.5
# How far the stair-step drops below the graph, as a share of the graph height.
STAIR_DROP = 0.06


def _finish_multiple_baseline(fig, ax, phase_lines, show_title, show_x, show_stair, stair_step_length):
    """Draw the phase change lines of one graph in a multiple baseline figure and
    set the area that is saved, so that the stacked graphs line up.

    Every graph in the stack is saved at the full figure width, so the plotting
    area sits at the same left and right position in every image no matter how
    wide the legend or the labels are. A graph below the top one (main title off)
    is cut off at the top of its plotting area, and its phase change line runs to
    that top edge. A graph with the stair-step is cut off at the bottom of its
    horizontal step. Stacked edge to edge, the step of one graph meets the phase
    change line of the graph below it.
    """
    fig_w, fig_h = fig.get_size_inches()
    renderer = fig.canvas.get_renderer()
    content = fig.get_tightbbox(renderer)  # inches, before the phase lines
    pad = 0.1
    axes_box = ax.get_position()  # figure fraction
    half_lw = PHASE_LINE_WIDTH / 72.0 / 2.0 / fig_h  # half a line width, figure fraction

    lower_graph = not show_title
    if lower_graph:
        top = max(axes_box.y1, content.y1 / fig_h)
    else:
        top = axes_box.y1
    step_y = axes_box.y0 - STAIR_DROP * axes_box.height

    # x in sessions, y in figure fraction, so the lines can leave the plotting area.
    trans = blended_transform_factory(ax.transData, fig.transFigure)
    x_right = ax.get_xlim()[1]
    for x_line in phase_lines:
        if show_stair:
            # Start at the far end of the step, where it meets the graph below,
            # so the dash pattern begins at the join. Never past the x-axis end.
            x_end = min(x_line + stair_step_length, x_right)
            xs = [x_end, x_line, x_line]
            ys = [step_y, step_y, top]
        else:
            xs = [x_line, x_line]
            ys = [top, axes_box.y0]
        ax.add_line(plt.Line2D(
            xs, ys, transform=trans, color="black", linestyle="--",
            linewidth=PHASE_LINE_WIDTH, clip_on=False, zorder=3,
        ))

    x0 = min(0.0, content.x0)
    x1 = max(fig_w, content.x1)
    if lower_graph:
        y1 = top * fig_h
    else:
        y1 = content.y1 + pad
    if show_stair and phase_lines and not show_x:
        y0 = (step_y - half_lw) * fig_h
    else:
        y0 = content.y0 - pad
    fig._scd_save_bbox = Bbox.from_extents(x0, y0, x1, y1)


def export_kwargs(fig):
    """Save settings for the graph: the fixed area for a multiple baseline
    graph, otherwise the tight crop the app has always used."""
    bbox = getattr(fig, "_scd_save_bbox", None)
    return {"bbox_inches": bbox if bbox is not None else "tight"}


# ---------------------------------------------------------------------------
# Saved settings
# ---------------------------------------------------------------------------

APP_NAME = "Single-Case Design Graph Generator"
SETTINGS_VERSION = 1
MAX_PHASES = 10
COLOR_MODES = ["Color", "Grayscale", "Custom"]

# Every setting outside the phases, with its default. The keys are the widget
# keys and also the names used in a saved settings file.
GLOBAL_DEFAULTS = {
    "num_phases": 2,
    "graph_title": "Single-Case Design Graph",
    "y_label": DEFAULT_Y_LABEL,
    "x_label": "Sessions",
    "y_min": 0.0,
    "y_max": 100.0,
    "y_tick": 10.0,
    "x_tick": 1.0,
    "use_max_x": False,
    "fixed_max_x": 30.0,
    "show_title": True,
    "show_phase_titles": True,
    "show_x": True,
    "show_legend": True,
    "offer_legend_downloads": True,
    "connect_gaps": False,
    "color_mode": "Color",
    "is_multiple_baseline": False,
    "extend_phase_lines": False,
    "stair_step_length": 3.0,
}

LOAD_ERROR = "Could not read that file. Make sure it is a settings file saved by this app."

# Session keys that belong to one phase or one measure.
PHASE_KEY_RE = re.compile(r"^(pt\d+|nm_\d+|mn\d+_\d+|md\d+_\d+|color_.*)$")
HEX_COLOR_RE = re.compile(r"^#[0-9a-fA-F]{6}$")


def measure_count(i):
    return max(1, int(st.session_state.get(f"nm_{i}", 1)))


def default_first_measure_name(state):
    return str(state.get("y_label", DEFAULT_Y_LABEL)).strip() or "Measure 1"


def build_settings(state):
    """Every choice on the page, as a dict ready to save as JSON."""
    out = {
        "app": APP_NAME,
        "version": SETTINGS_VERSION,
        "saved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    for k, default in GLOBAL_DEFAULTS.items():
        if k == "num_phases":
            continue
        out[k] = state.get(k, default)

    first_name = default_first_measure_name(state)
    phases = []
    num_phases = int(state.get("num_phases", GLOBAL_DEFAULTS["num_phases"]))
    for i in range(num_phases):
        n = max(1, int(state.get(f"nm_{i}", 1)))
        measures = []
        for j in range(n):
            if n >= 2:
                fallback = first_name if j == 0 else f"Measure {j + 1}"
                name = str(state.get(f"mn{i}_{j}", fallback)).strip() or f"Measure {j + 1}"
            else:
                name = first_name
            measures.append({"name": name, "data": str(state.get(f"md{i}_{j}", ""))})
        phases.append({"title": str(state.get(f"pt{i}", f"Phase {i + 1}")), "measures": measures})
    out["phases"] = phases

    colors = {}
    for p in phases:
        for m in p["measures"]:
            key = f"color_{m['name']}"
            if key in state:
                colors[m["name"]] = state[key]
    out["custom_colors"] = colors
    return out


def _as_bool(v, default):
    return v if isinstance(v, bool) else default


def _as_float(v, default, lowest=None):
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        try:
            v = float(str(v).strip())
        except (TypeError, ValueError):
            return default
    v = float(v)
    if not np.isfinite(v):
        return default
    if lowest is not None and v < lowest:
        return lowest
    return v


def _as_text(v, default):
    if v is None:
        return default
    if isinstance(v, (str, int, float)) and not isinstance(v, bool):
        return str(v)
    return default


def _as_data_text(v):
    """A data box entry. A list of numbers from a hand-made file is accepted
    and written the way a student would type it."""
    if isinstance(v, list):
        parts = []
        for x in v:
            if x is None or (isinstance(x, float) and np.isnan(x)):
                parts.append("-")
            else:
                parts.append(str(x))
        return ", ".join(parts)
    return _as_text(v, "")


def settings_to_state(data):
    """Turn a loaded settings file into session state values.

    Raises ValueError when the file is not a settings file from this app.
    A setting missing from the file, or one that cannot be read, gets its
    default, so a file saved before a later change still loads.
    Returns (updates, note).
    """
    if not isinstance(data, dict):
        raise ValueError("not an object")
    app = data.get("app")
    if app is not None and app != APP_NAME:
        raise ValueError("another app")
    if app is None and "phases" not in data:
        raise ValueError("not a settings file")

    updates = {}
    for k, default in GLOBAL_DEFAULTS.items():
        if k == "num_phases":
            continue
        v = data.get(k, default)
        if isinstance(default, bool):
            updates[k] = _as_bool(v, default)
        elif isinstance(default, float):
            lowest = {"fixed_max_x": 1.0, "stair_step_length": 0.0}.get(k)
            updates[k] = _as_float(v, default, lowest)
        else:
            updates[k] = _as_text(v, default)
    if updates["color_mode"] not in COLOR_MODES:
        updates["color_mode"] = GLOBAL_DEFAULTS["color_mode"]

    phases = data.get("phases")
    if phases is None:
        phases = [{} for _ in range(GLOBAL_DEFAULTS["num_phases"])]
    if not isinstance(phases, list):
        raise ValueError("phases is not a list")
    if not phases:
        phases = [{}]
    note = ""
    if len(phases) > MAX_PHASES:
        note = (f" This file has {len(phases)} phases. The app shows up to {MAX_PHASES}, "
                f"so the first {MAX_PHASES} were loaded.")
        phases = phases[:MAX_PHASES]
    updates["num_phases"] = len(phases)

    first_name = str(updates["y_label"]).strip() or "Measure 1"
    for i, p in enumerate(phases):
        if not isinstance(p, dict):
            p = {}
        updates[f"pt{i}"] = _as_text(p.get("title"), f"Phase {i + 1}")
        measures = p.get("measures")
        if not isinstance(measures, list) or not measures:
            measures = [{}]
        n = len(measures)
        updates[f"nm_{i}"] = n
        for j, m in enumerate(measures):
            if not isinstance(m, dict):
                m = {}
            if n >= 2:
                fallback = first_name if j == 0 else f"Measure {j + 1}"
                updates[f"mn{i}_{j}"] = _as_text(m.get("name"), fallback)
            updates[f"md{i}_{j}"] = _as_data_text(m.get("data", ""))

    colors = data.get("custom_colors")
    if isinstance(colors, dict):
        for name, hexval in colors.items():
            if isinstance(name, str) and isinstance(hexval, str) and HEX_COLOR_RE.match(hexval):
                updates[f"color_{name}"] = hexval
    return updates, note


def _load_settings(uploader_key):
    f = st.session_state.get(uploader_key)
    if f is None:
        return
    try:
        data = json.loads(f.getvalue().decode("utf-8-sig"))
        updates, note = settings_to_state(data)
    except Exception:
        st.session_state["_load_msg"] = ("error", LOAD_ERROR)
        return
    for k in list(st.session_state.keys()):
        if PHASE_KEY_RE.match(str(k)):
            del st.session_state[k]
    for k, v in updates.items():
        st.session_state[k] = v
    # A fresh, empty upload box, so the same file can be loaded again later.
    st.session_state["_uploader_n"] = st.session_state.get("_uploader_n", 0) + 1
    st.session_state["_load_msg"] = ("success", "Your saved settings are loaded." + note)
    st.session_state["_draw_after_load"] = True


def _add_measure(i):
    n = measure_count(i)
    st.session_state.pop(f"mn{i}_{n}", None)
    st.session_state.pop(f"md{i}_{n}", None)
    st.session_state[f"nm_{i}"] = n + 1


def _remove_measure(i, j):
    n = measure_count(i)
    if n <= 1:
        return
    ss = st.session_state
    for k in range(j, n - 1):
        for prefix in ("mn", "md"):
            nxt = f"{prefix}{i}_{k + 1}"
            if nxt in ss:
                ss[f"{prefix}{i}_{k}"] = ss[nxt]
            else:
                ss.pop(f"{prefix}{i}_{k}", None)
    ss.pop(f"mn{i}_{n - 1}", None)
    ss.pop(f"md{i}_{n - 1}", None)
    ss[f"nm_{i}"] = n - 1
    if n - 1 == 1:
        # One measure left: it is named from the Y-axis label again.
        ss.pop(f"mn{i}_0", None)


def _settings_file_name(typed, graph_title):
    name = (typed or "").strip() or f"{(graph_title or 'graph').strip() or 'graph'} settings"
    name = re.sub(r'[\\/:*?"<>|]+', "-", name).strip() or "graph settings"
    if not name.lower().endswith(".json"):
        name += ".json"
    return name


def settings_section():
    st.subheader("Save or Load Your Settings")
    st.markdown(
        "Save your settings as a file, and load that file later to fill in every box "
        "on this page as you left it."
    )
    col_save, col_load = st.columns(2)
    with col_save:
        typed = st.text_input(
            "Save settings as",
            key="_save_name",
            placeholder=_settings_file_name("", st.session_state.get("graph_title")),
        )
        st.download_button(
            "Save my settings",
            data=json.dumps(build_settings(st.session_state), indent=2),
            file_name=_settings_file_name(typed, st.session_state.get("graph_title")),
            mime="application/json",
            on_click="ignore",
        )
    with col_load:
        uploader_key = f"_settings_upload_{st.session_state.get('_uploader_n', 0)}"
        st.file_uploader(
            "Load saved settings",
            type=["json"],
            key=uploader_key,
            on_change=_load_settings,
            args=(uploader_key,),
        )
    msg = st.session_state.pop("_load_msg", None)
    if msg:
        kind, text = msg
        if kind == "error":
            st.error(text)
        else:
            st.success(text)


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

def main():
    st.title("Single-Case Design Graph Generator (Advanced + Multiple Baseline Mode)")

    for k, v in GLOBAL_DEFAULTS.items():
        st.session_state.setdefault(k, v)

    settings_section()

    num_phases = st.number_input("How many phases?", min_value=1, max_value=MAX_PHASES, key="num_phases")

    phase_titles = []
    phase_measures = []  # per phase: [(measure_name, data_list), ...]

    # The Y-axis label box sits further down the page, so read its current
    # value from session state. It names a measure that has no name box.
    default_m1_name = default_first_measure_name(st.session_state)

    for i in range(num_phases):
        st.subheader(f"Phase {i+1} Settings")
        st.session_state.setdefault(f"pt{i}", f"Phase {i+1}")
        ptitle = st.text_input(f"Title for Phase {i+1}", key=f"pt{i}")
        phase_titles.append(ptitle)

        # The measure name boxes appear only when the phase has more than one
        # measure. Otherwise the measure is named from the Y-axis label.
        n = measure_count(i)
        measures = []
        for j in range(n):
            if n >= 2:
                st.session_state.setdefault(
                    f"mn{i}_{j}", default_m1_name if j == 0 else f"Measure {j+1}"
                )
                mname = st.text_input(f"Name for Measure {j+1} in {ptitle}", key=f"mn{i}_{j}").strip() or f"Measure {j+1}"
                data_label = mname
            else:
                mname = default_m1_name
                data_label = ptitle
            mdata = parse_series(st.text_input(f"Data for {data_label}", key=f"md{i}_{j}"))
            st.caption(DATA_BOX_HELP)
            if n >= 2:
                st.button(
                    f"Remove {mname} from {ptitle}",
                    key=f"rm{i}_{j}",
                    on_click=_remove_measure,
                    args=(i, j),
                )
            measures.append((mname, mdata))

        st.button(f"Add another measure to {ptitle}", key=f"add_{i}", on_click=_add_measure, args=(i,))
        st.caption(
            "Add a measure only for a phase with more than one line, as in an alternating treatments design. "
            "Each line then needs its own name for the legend."
        )

        names = [m[0] for m in measures]
        repeated = [nm for k, nm in enumerate(names) if nm in names[:k]]
        if repeated:
            shown = repeated[0]
            if n == 2:
                st.warning(
                    f"Both measures in {ptitle} are named {shown}, so they will share one color, "
                    "one marker, and one legend entry. Give each measure its own name."
                )
            else:
                st.warning(
                    f"More than one measure in {ptitle} is named {shown}, so those measures will share "
                    "one color, one marker, and one legend entry. Give each measure its own name."
                )

        phase_measures.append(measures)

    st.header("Axis Settings")
    graph_title = st.text_input("Graph Title", key="graph_title")
    y_label = st.text_input("Dependent variable (Y-axis label)", key="y_label")
    x_label = st.text_input("Unit of measurement (X-axis label)", key="x_label")
    y_min = st.number_input("Minimum Y value", key="y_min")
    y_max = st.number_input("Maximum Y value", key="y_max")
    y_tick = st.number_input("Y tick interval", key="y_tick")
    x_tick = st.number_input("X tick interval", key="x_tick")

    st.subheader("X-axis Range Control")
    st.markdown(
        "Use this only when this graph is one graph in a multiple baseline figure. "
        "Leave it off for a single graph.\n\n"
        "In a multiple baseline figure, the graphs are stacked from top to bottom, and every graph "
        "in the stack needs the same maximum X value so the sessions line up vertically. "
        "Enter the highest total number of sessions for any participant, counting every phase, "
        "and enter that same number on every graph in the stack. For example, if three participants "
        "had 18, 22, and 25 total sessions, enter 25 on all three graphs."
    )
    use_max_x = st.checkbox("Set a fixed maximum X value for alignment", key="use_max_x")
    fixed_max_x = None
    if use_max_x:
        st.session_state.setdefault("fixed_max_x", GLOBAL_DEFAULTS["fixed_max_x"])
        fixed_max_x = st.number_input("Maximum X value", min_value=1.0, step=1.0, key="fixed_max_x")

    st.header("Advanced Graph Controls")
    st.markdown(
        "For a single graph, leave every box below on.\n\n"
        "For a multiple baseline figure, make one graph for each participant, behavior, or setting, "
        "then stack them from top to bottom. Use the same settings on every graph, except:\n\n"
        "- **Top graph:** turn on the main graph title. Turn off the x-axis.\n"
        "- **Every middle graph:** turn off the main graph title and the x-axis.\n"
        "- **Bottom graph:** turn off the main graph title. Turn on the x-axis.\n\n"
        "The x-axis box controls the axis line, the tick marks, the session numbers, and the "
        "axis label together, so only the bottom graph shows session numbers."
    )
    show_title = st.checkbox(
        "Show main graph title (turn this off for every graph in a multiple baseline except the top one)",
        key="show_title",
    )
    show_phase_titles = st.checkbox("Show phase titles above each phase start", key="show_phase_titles")
    show_x = st.checkbox(
        "Show the x-axis (turn this off for every graph in a multiple baseline except the bottom one)",
        key="show_x",
    )
    show_legend = st.checkbox("Show legend on graph", key="show_legend")
    offer_legend_downloads = st.checkbox("Offer legend as separate downloads", key="offer_legend_downloads")
    connect_gaps = st.checkbox("Connect each measure's points across sessions with no data", key="connect_gaps")
    st.caption(
        "Leave this off for most graphs, so the line breaks at a session with no data point. "
        "Turn it on for an alternating treatments design, where each condition's points are joined "
        "across the sessions when the other condition ran. To enter an alternating treatments design, "
        "add a second measure to the phase and type a hyphen for each session when that condition did "
        "not run, as in 5, -, 6, - for one condition and -, 3, -, 4 for the other."
    )

    st.subheader("Color Options")
    color_mode = st.radio("Select color mode:", COLOR_MODES, key="color_mode")

    custom_colors = {}
    if color_mode == "Custom":
        for measure_name in distinct_measure_names(phase_measures):
            st.session_state.setdefault(f"color_{measure_name}", "#000000")
            custom_colors[measure_name] = st.color_picker(
                f"Select color for {measure_name}", key=f"color_{measure_name}"
            )

    st.header("Multiple Baseline Options")
    is_multiple_baseline = st.checkbox("This graph is part of a multiple baseline figure", key="is_multiple_baseline")
    extend_phase_lines = False
    stair_step_length = 0.0
    if is_multiple_baseline:
        st.session_state.setdefault("extend_phase_lines", GLOBAL_DEFAULTS["extend_phase_lines"])
        extend_phase_lines = st.checkbox("Show stair-step extension at phase changes", key="extend_phase_lines")
        if extend_phase_lines:
            st.session_state.setdefault("stair_step_length", GLOBAL_DEFAULTS["stair_step_length"])
            stair_step_length = st.number_input(
                "Horizontal extension length (in sessions)", min_value=0.0, step=0.5, key="stair_step_length"
            )

    # After a settings file is loaded, the graph is drawn once without a click.
    generate = st.button("Generate Graph")
    if st.session_state.pop("_draw_after_load", False):
        generate = True

    if generate:
        fig, ax, handles, labels = build_figure(
            phase_titles,
            phase_measures,
            graph_title=graph_title,
            y_label=y_label,
            x_label=x_label,
            y_min=y_min,
            y_max=y_max,
            y_tick=y_tick,
            x_tick=x_tick,
            use_max_x=use_max_x,
            fixed_max_x=fixed_max_x,
            show_title=show_title,
            show_phase_titles=show_phase_titles,
            show_x=show_x,
            show_legend=show_legend,
            connect_gaps=connect_gaps,
            color_mode=color_mode,
            custom_colors=custom_colors,
            is_multiple_baseline=is_multiple_baseline,
            extend_phase_lines=extend_phase_lines,
            stair_step_length=stair_step_length,
        )

        # Drawn the same way st.pyplot draws it (PNG at 200 dpi, full width).
        # Passing save settings to st.pyplot itself now prints a notice on the page.
        shown_buf = io.BytesIO()
        fig.savefig(shown_buf, format="png", dpi=200, **export_kwargs(fig))
        st.image(shown_buf.getvalue(), width="stretch")

        png_buf = io.BytesIO()
        fig.savefig(png_buf, format="png", transparent=True, **export_kwargs(fig))
        st.download_button(
            label="Download PNG",
            data=png_buf.getvalue(),
            file_name=f"{graph_title.replace(' ', '_')}.png",
            mime="image/png",
        )

        svg_buf = io.BytesIO()
        fig.savefig(svg_buf, format="svg", transparent=True, **export_kwargs(fig))
        st.download_button(
            label="Download SVG",
            data=svg_buf.getvalue(),
            file_name=f"{graph_title.replace(' ', '_')}.svg",
            mime="image/svg+xml",
        )

        if offer_legend_downloads and labels:
            n_items = len(labels)
            leg_fig = plt.figure(figsize=(4.0, max(1.0, 0.45 * n_items)))
            leg_fig.legend(handles, labels, loc="center", frameon=False, ncol=1)
            leg_fig.canvas.draw()

            leg_png = io.BytesIO()
            leg_fig.savefig(leg_png, format="png", bbox_inches="tight", transparent=True)
            st.download_button(
                label="Download Legend PNG",
                data=leg_png.getvalue(),
                file_name=f"{graph_title.replace(' ', '_')}_legend.png",
                mime="image/png",
            )

            leg_svg = io.BytesIO()
            leg_fig.savefig(leg_svg, format="svg", bbox_inches="tight", transparent=True)
            st.download_button(
                label="Download Legend SVG",
                data=leg_svg.getvalue(),
                file_name=f"{graph_title.replace(' ', '_')}_legend.svg",
                mime="image/svg+xml",
            )

            plt.close(leg_fig)

    st.markdown(
        """
        <div style="text-align: center; margin-top: 2em;">
            <a href="https://www.buymeacoffee.com/seanaevelyd" target="_blank">
                <img
                    src="https://cdn.buymeacoffee.com/buttons/v2/default-yellow.png"
                    alt="Buy Me A Coffee"
                    style="height: 60px; width: 217px;"
                >
            </a>
        </div>
        """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
