import io
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.transforms import blended_transform_factory
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

    default_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
    grayscale_colors = ["#000000", "#555555", "#888888", "#AAAAAA", "#CCCCCC", "#EEEEEE"]
    markers = ["o", "s", "D", "^", "v", "P"]

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
            x_line = start_x + L - 0.5
            ax.axvline(x=x_line, color="black", linestyle="--", linewidth=1.5, zorder=3)

            if is_multiple_baseline and extend_phase_lines and stair_step_length > 0:
                bt = blended_transform_factory(ax.transData, ax.transAxes)
                drop_axes = 0.06
                ax.vlines(
                    x_line, 0.0, -drop_axes,
                    colors="black", linestyles="--", linewidth=1.8,
                    transform=bt, clip_on=False, zorder=4
                )
                ax.hlines(
                    -drop_axes, x_line, x_line + stair_step_length,
                    colors="black", linestyles="--", linewidth=1.8,
                    transform=bt, clip_on=False, zorder=4
                )
                plotted_xmax = max(plotted_xmax, x_line + stair_step_length + 0.5)

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

    return fig, ax, handles, labels


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

def main():
    st.title("Single-Case Design Graph Generator (Advanced + Multiple Baseline Mode)")

    num_phases = st.number_input("How many phases?", min_value=1, max_value=5, value=2)

    phase_titles = []
    phase_measures = []  # per phase: [(measure_name, data_list), ...]

    # The Y-axis label box sits further down the page, so read its current
    # value from session state. It names a measure that has no name box.
    default_m1_name = str(st.session_state.get("y_label", DEFAULT_Y_LABEL)).strip() or "Measure 1"

    for i in range(num_phases):
        st.subheader(f"Phase {i+1} Settings")
        ptitle = st.text_input(f"Title for Phase {i+1}", value=f"Phase {i+1}", key=f"pt{i}")
        phase_titles.append(ptitle)

        # The measure name boxes appear only when the phase has a second
        # measure. Otherwise the measure is named from the Y-axis label.
        two_measures = bool(st.session_state.get(f"add2_{i}", False))
        if two_measures:
            if f"m1n{i}" not in st.session_state:
                st.session_state[f"m1n{i}"] = default_m1_name
            m1_name = st.text_input(f"Name for Measure 1 in {ptitle}", key=f"m1n{i}").strip() or "Measure 1"
            m1_label = m1_name
        else:
            m1_name = default_m1_name
            m1_label = ptitle
        m1_data = parse_series(st.text_input(f"Data for {m1_label}", key=f"m1d{i}"))
        st.caption(DATA_BOX_HELP)

        add_second = st.checkbox(f"Add second measure for {ptitle}", key=f"add2_{i}")
        st.caption(
            "Check this only for a phase with two lines, as in an alternating treatments design. "
            "Each line then needs its own name for the legend."
        )
        measures = [(m1_name, m1_data)]
        if add_second and two_measures:
            if f"m2n{i}" not in st.session_state:
                st.session_state[f"m2n{i}"] = "Measure 2"
            m2_name = st.text_input(f"Name for Measure 2 in {ptitle}", key=f"m2n{i}").strip() or "Measure 2"
            m2_data = parse_series(st.text_input(f"Data for {m2_name}", key=f"m2d{i}"))
            st.caption(DATA_BOX_HELP)
            if m2_name == m1_name:
                st.warning(
                    f"Both measures in {ptitle} are named {m1_name}, so they will share one color, "
                    "one marker, and one legend entry. Give each measure its own name."
                )
            measures.append((m2_name, m2_data))

        phase_measures.append(measures)

    st.header("Axis Settings")
    graph_title = st.text_input("Graph Title", "Single-Case Design Graph")
    y_label = st.text_input("Dependent variable (Y-axis label)", DEFAULT_Y_LABEL, key="y_label")
    x_label = st.text_input("Unit of measurement (X-axis label)", "Sessions")
    y_min = st.number_input("Minimum Y value", value=0.0)
    y_max = st.number_input("Maximum Y value", value=100.0)
    y_tick = st.number_input("Y tick interval", value=10.0)
    x_tick = st.number_input("X tick interval", value=1.0)

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
    use_max_x = st.checkbox("Set a fixed maximum X value for alignment", value=False)
    fixed_max_x = None
    if use_max_x:
        fixed_max_x = st.number_input("Maximum X value", min_value=1.0, value=30.0, step=1.0)

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
        value=True,
    )
    show_phase_titles = st.checkbox("Show phase titles above each phase start", value=True)
    show_x = st.checkbox(
        "Show the x-axis (turn this off for every graph in a multiple baseline except the bottom one)",
        value=True,
    )
    show_legend = st.checkbox("Show legend on graph", value=True)
    offer_legend_downloads = st.checkbox("Offer legend as separate downloads", value=True)
    connect_gaps = st.checkbox("Connect each measure's points across sessions with no data", value=False)
    st.caption(
        "Leave this off for most graphs, so the line breaks at a session with no data point. "
        "Turn it on for an alternating treatments design, where each condition's points are joined "
        "across the sessions when the other condition ran. To enter an alternating treatments design, "
        "add a second measure to the phase and type a hyphen for each session when that condition did "
        "not run, as in 5, -, 6, - for one condition and -, 3, -, 4 for the other."
    )

    st.subheader("Color Options")
    color_mode = st.radio("Select color mode:", ["Color", "Grayscale", "Custom"], index=0)

    custom_colors = {}
    if color_mode == "Custom":
        for measure_name in distinct_measure_names(phase_measures):
            custom_colors[measure_name] = st.color_picker(
                f"Select color for {measure_name}", "#000000", key=f"color_{measure_name}"
            )

    st.header("Multiple Baseline Options")
    is_multiple_baseline = st.checkbox("This graph is part of a multiple baseline figure", value=False)
    extend_phase_lines = False
    stair_step_length = 0.0
    if is_multiple_baseline:
        extend_phase_lines = st.checkbox("Show stair-step extension at phase changes", value=False)
        if extend_phase_lines:
            stair_step_length = st.number_input("Horizontal extension length (in sessions)", min_value=0.0, value=3.0, step=0.5)

    if st.button("Generate Graph"):
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

        st.pyplot(fig)

        png_buf = io.BytesIO()
        fig.savefig(png_buf, format="png", bbox_inches="tight", transparent=True)
        st.download_button(
            label="Download PNG",
            data=png_buf.getvalue(),
            file_name=f"{graph_title.replace(' ', '_')}.png",
            mime="image/png",
        )

        svg_buf = io.BytesIO()
        fig.savefig(svg_buf, format="svg", bbox_inches="tight", transparent=True)
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
