import re
import io
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.transforms import blended_transform_factory
from matplotlib.ticker import MultipleLocator
import streamlit as st

st.title("Single-Case Design Graph Generator (Advanced + Multiple Baseline Mode)")

# How many phases
num_phases = st.number_input("How many phases?", min_value=1, max_value=5, value=2)

phase_titles = []
phase_measures = []  # per phase: [(measure_name, data_list), ...]

def parse_series(s: str):
    """Parse numbers with missing markers, allowing comma or whitespace separators."""
    if not s:
        return []
    tokens = re.split(r"[,\s]+", s.strip())
    out = []
    for t in tokens:
        tt = t.strip().lower()
        if tt in ("", "x", "-", "na"):
            out.append(np.nan)
        else:
            try:
                out.append(float(tt))
            except ValueError:
                st.error(f"Invalid entry '{t}'. Use numbers or 'x'/'-'/'na' for missing.")
                return []
    return out

# Phase inputs
for i in range(num_phases):
    st.subheader(f"Phase {i+1} Settings")
    ptitle = st.text_input(f"Title for Phase {i+1}", value=f"Phase {i+1}", key=f"pt{i}")
    phase_titles.append(ptitle)

    m1_name = st.text_input(f"Name for Measure 1 in {ptitle}", value=f"Measure 1 ({ptitle})", key=f"m1n{i}")
    m1_data = parse_series(st.text_input(f"Data for {m1_name} (comma or space separated)", key=f"m1d{i}"))

    add_second = st.checkbox(f"Add second measure for {ptitle}?", key=f"add2_{i}")
    measures = [(m1_name, m1_data)]
    if add_second:
        m2_name = st.text_input(f"Name for Measure 2 in {ptitle}", value=f"Measure 2 ({ptitle})", key=f"m2n{i}")
        m2_data = parse_series(st.text_input(f"Data for {m2_name} (comma or space separated)", key=f"m2d{i}"))
        measures.append((m2_name, m2_data))

    phase_measures.append(measures)

# Axis labels and scaling
st.header("Axis Settings")
graph_title = st.text_input("Graph Title", "Single-Case Design Graph")
y_label = st.text_input("Dependent variable (Y-axis label)", "% correct responses")
x_label = st.text_input("Unit of measurement (X-axis label)", "Sessions")
y_min = st.number_input("Minimum Y value", value=0.0)
y_max = st.number_input("Maximum Y value", value=100.0)
y_tick = st.number_input("Y tick interval", value=10.0)
x_tick = st.number_input("X tick interval", value=1.0)

# Optional Max X control for alignment across stacked graphs
st.subheader("X-axis Range Control")
use_max_x = st.checkbox("Set a fixed maximum X value for alignment", value=False)
fixed_max_x = None
if use_max_x:
    fixed_max_x = st.number_input("Maximum X value", min_value=1.0, value=30.0, step=1.0)

# Advanced display controls
st.header("Advanced Graph Controls")
show_title = st.checkbox("Show main graph title", value=True)
show_phase_titles = st.checkbox("Show phase titles above each phase start", value=True)
show_x_axis = st.checkbox("Show X-axis line", value=True)
show_x_ticks = st.checkbox("Show X tick marks", value=True)
show_x_tick_labels = st.checkbox("Show X tick labels", value=True)
show_x_label = st.checkbox("Show X-axis label text", value=True)
show_legend = st.checkbox("Show legend on graph", value=True)
offer_legend_downloads = st.checkbox("Offer legend as separate downloads", value=True)

# Color options
st.subheader("Color Options")
color_mode = st.radio("Select color mode:", ["Color", "Grayscale", "Custom"], index=0)

custom_colors = []
if color_mode == "Custom":
    for i, phase in enumerate(phase_measures):
        for j, (measure_name, _) in enumerate(phase):
            color = st.color_picker(f"Select color for {measure_name}", "#000000", key=f"color_{i}_{j}")
            custom_colors.append(color)

# Multiple Baseline options
st.header("Multiple Baseline Options")
is_multiple_baseline = st.checkbox("This graph is part of a multiple baseline figure", value=False)
extend_phase_lines = False
stair_step_length = 0.0
if is_multiple_baseline:
    extend_phase_lines = st.checkbox("Show stair-step extension at phase changes", value=False)
    if extend_phase_lines:
        stair_step_length = st.number_input("Horizontal extension length (in sessions)", min_value=0.0, value=3.0, step=0.5)

# Generate
if st.button("Generate Graph"):
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
    elif color_mode == "Custom" and custom_colors:
        palette = custom_colors
    else:
        palette = default_colors

    # Precompute phase lengths and starts
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

    # Plot
    color_index = 0
    for idx, (ptitle, measures) in enumerate(zip(phase_titles, phase_measures)):
        start_x = phase_starts[idx]
        L = phase_lengths[idx]

        # Measures
        for j, (mname, data) in enumerate(measures):
            if not data:
                continue
            x_vals = np.arange(start_x, start_x + len(data))
            color = palette[color_index % len(palette)]
            color_index += 1
            ax.plot(
                x_vals, data,
                color=color,
                marker=markers[(idx + j) % len(markers)],
                label=mname,
                linewidth=2,
                markersize=6,
            )
            if len(x_vals) > 0:
                plotted_xmax = max(plotted_xmax, x_vals[-1] + 0.5)

        # Phase title
        if show_phase_titles and L > 0:
            ax.text(
                start_x, y_top, ptitle,
                ha="left", va="bottom",
                fontsize=10, fontweight="bold", wrap=True
            )

        # Phase change separator
        if idx < len(phase_titles) - 1 and L > 0:
            x_line = start_x + L - 0.5
            ax.axvline(x=x_line, color="black", linestyle="--", linewidth=1.5, zorder=3)

            # Optional stair-step
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

    # Formatting
    if show_title:
        ax.set_title(graph_title, fontsize=13, pad=20)
    ax.set_ylabel(y_label)
    if show_x_label:
        ax.set_xlabel(x_label)

    ax.set_ylim(y_min, y_max + (y_max - y_min) * 0.1)
    ax.set_yticks(np.arange(y_min, y_max + 1e-9, y_tick))
    ax.xaxis.set_major_locator(MultipleLocator(base=x_tick))

    ax.tick_params(axis="x", which="both", bottom=show_x_ticks, top=False)
    if not show_x_tick_labels:
        ax.tick_params(axis="x", labelbottom=False)
    if not show_x_axis:
        ax.spines["bottom"].set_visible(False)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # X-limits
    if use_max_x and fixed_max_x is not None:
        ax.set_xlim(0.5, max(fixed_max_x + 0.5, plotted_xmax))
    else:
        ax.set_xlim(0.5, max(plotted_xmax, 2.5))

    # Legend on the main graph
    handles, labels = ax.get_legend_handles_labels()
    if show_legend and labels:
        ax.legend(frameon=False, loc="center left", bbox_to_anchor=(1.02, 0.5))

    st.pyplot(fig)

    # Main graph downloads, transparent background
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

    # Legend-only downloads
    if offer_legend_downloads and labels:
        # Create a separate, compact figure that only contains the legend
        # Size adjusts with number of items
        n_items = len(labels)
        leg_fig = plt.figure(figsize=(4.0, max(1.0, 0.45 * n_items)))
        # Place legend at center
        leg = leg_fig.legend(
            handles, labels,
            loc="center", frameon=False, ncol=1
        )
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
