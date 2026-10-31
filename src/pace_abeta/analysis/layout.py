"""Fixed physical axes geometry and role-based manuscript typography."""
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
from .fonts import FONT_FAMILY

LAYOUT = json.loads(Path(__file__).with_name("figure_layout.json").read_text())


def panel_letter(fig, letter, width, height, *, font_pt=None):
    """Use the same physical label anchor for every result panel."""
    style = LAYOUT["panel_letter"]
    fig.text(style["x_mm"] / width,
             (height - style["baseline_from_top_mm"]) / height,
             letter, fontfamily=FONT_FAMILY,
             fontsize=style["font_pt"] if font_pt is None else font_pt,
             fontweight="bold", ha="left", va="baseline")


def plot_signature(fig):
    """Exact statistical arrays and visual layers, independent of output geometry."""
    result=[]
    for ax in fig.axes:
        result.append(dict(
            lines=[(np.asarray(l.get_xdata()).tolist(),np.asarray(l.get_ydata()).tolist(),
                    str(l.get_color()),l.get_linestyle(),l.get_alpha()) for l in ax.lines],
            collections=[(np.asarray(c.get_offsets()).tolist(),c.get_facecolors().tolist(),
                          c.get_edgecolors().tolist(),c.get_alpha()) for c in ax.collections],
            patches=[(p.get_facecolor(),p.get_edgecolor(),p.get_alpha(),
                      p.get_path().vertices.tolist(),p.get_patch_transform().get_matrix().tolist()) for p in ax.patches],
            annotations=[t.get_text() for t in ax.texts]))
    return json.dumps(result,sort_keys=True,default=float)

# Presentation-only naming; training rank normalization is part of the score definition.
# Internal table keys are independent of visible method labels.
METHOD_LABEL = r'PACE-A$\beta$'
METHOD_RENAMES = {
    'DAPS': METHOD_LABEL,
    'Rank-calibrated DAPS score': METHOD_LABEL + ' score',
    'Pseudotime': METHOD_LABEL + ' score',
}

def rename_visible_method(fig):
    changes = []
    # Keep normal letters and beta in the existing Arial family. Mathtext is
    # used only for beta; no font size, weight, axis geometry or data is changed.
    plt.rcParams.update({'mathtext.fontset': 'custom', 'mathtext.rm': FONT_FAMILY,
                         'mathtext.it': FONT_FAMILY + ':italic', 'mathtext.bf': FONT_FAMILY + ':bold'})
    for ax in fig.axes:
        for axis in (ax.xaxis, ax.yaxis):
            label = axis.label
            before = label.get_text()
            if before in METHOD_RENAMES:
                label.set_text(METHOD_RENAMES[before])
                changes.append({'before': before, 'after': METHOD_RENAMES[before]})
            ticks = axis.get_ticklabels()
            old = [t.get_text() for t in ticks]
            renamed = [METHOD_RENAMES.get(text, text) for text in old]
            if renamed != old:
                axis.set_ticks(axis.get_ticklocs(), labels=renamed)
                changes.extend({'before': a, 'after': b} for a, b in zip(old, renamed) if a != b)
    return changes

def export(fig, name, figure_dir):
    FIG = Path(figure_dir)
    RAW = FIG / "original_style"
    RAW.mkdir(parents=True, exist_ok=True)
    visible_label_changes = rename_visible_method(fig)
    fig.canvas.draw()
    # Export the plot before its final-size typography adjustments.
    fig.savefig(RAW/f'{name}.svg',dpi=300)
    fig.savefig(RAW/f'{name}.pdf',dpi=300)
    fig.savefig(RAW/f'{name}.png',dpi=160)
    before=plot_signature(fig)
    ax=fig.axes[0]
    original_yticks=ax.get_yticks().copy()
    orig_width,orig_height=fig.get_size_inches()*25.4
    is_comp=name.startswith('figure2')
    geometry=LAYOUT['figure2' if is_comp else 'figure3']
    width,height=geometry['canvas_mm']
    left,bottom,aw,ah=geometry['axes_rect_mm']
    # Preserve the existing stroke/marker sizes when adding panel whitespace.
    scale=(178/3 if is_comp else 74)/orig_width
    # Scale strokes and markers to the standalone panel. Font sizes below
    # are panel points; manuscript scaling applies uniformly to the figure.
    for line in ax.lines:
        line.set_linewidth(line.get_linewidth()*scale)
        line.set_markersize(line.get_markersize()*scale)
        line.set_markeredgewidth(line.get_markeredgewidth()*scale)
    for c in ax.collections:
        c.set_linewidths(c.get_linewidths()*scale)
        if hasattr(c,'get_sizes'):c.set_sizes(c.get_sizes()*scale**2)
    for p in ax.patches:p.set_linewidth(p.get_linewidth()*scale)
    for s in ax.spines.values():s.set_linewidth(s.get_linewidth()*scale)
    fig.set_size_inches(width/25.4,height/25.4,forward=False)
    fig.set_layout_engine(None)
    ax.set_position([left/width,bottom/height,aw/width,ah/height])
    if is_comp:
        lo,hi=ax.get_ylim();ticks=original_yticks
        ax.set_yticks([v for v in ticks if lo-1e-9<=v<=hi+1e-9]);ax.set_ylim(lo,hi)
    else:
        # The same coordinates represent the same physical displacement in A/B.
        ax.set_xlim(-.04,1.04);ax.set_ylim(-75,225)
        ax.set_xticks(np.arange(0,1.01,.2));ax.set_yticks(np.arange(-50,201,50))
    fonts = geometry.get('fonts_pt', {})
    tick_size = fonts.get('tick', 8)
    axis_title_size = fonts.get('axis_title', 9)
    panel_title_size = fonts.get('panel_title', 10)
    ax.tick_params(axis='both',labelsize=tick_size,pad=2.5,width=.55,length=2.5)
    for t in ax.get_xticklabels()+ax.get_yticklabels():
        t.set_fontfamily(FONT_FAMILY);t.set_fontsize(tick_size)
    ax.xaxis.label.set_fontsize(7.5 if is_comp else axis_title_size)
    ax.yaxis.label.set_fontsize(axis_title_size)
    ax.set_title(ax.get_title(),fontsize=panel_title_size,pad=4)
    ax.title.set_position((.5,1.02))
    ax.title.set_y(1.02)
    ax.title.set_va('bottom')
    if not is_comp:
        ax.xaxis.set_label_coords(.5,-6.7/ah)
    y_title_x = 3.7 if is_comp else 4.2
    ax.yaxis.set_label_coords((y_title_x-left)/aw,.5)
    annotation_size=6.5 if name in ('figure2a', 'figure2c') else fonts.get('annotation', 7.5)
    for t in ax.texts:
        t.set_fontsize(annotation_size);t.set_fontweight('normal')
    if is_comp and name!='figure2a':
        ax.set_xlabel('')
        for t in ax.get_xticklabels():
            t.set_rotation(40);t.set_ha('right');t.set_rotation_mode('anchor')
    if name=='figure2a':
        # Keep Diagnosis small and directly below its horizontal tick labels.
        # B/C use their method tick names without a redundant axis title.
        fig.canvas.draw();ren=fig.canvas.get_renderer()
        tick_bottom=min(t.get_window_extent(ren).y0 for t in ax.get_xticklabels()
                        if t.get_visible() and t.get_text())
        label_top=tick_bottom-1.0/25.4*fig.dpi
        y=ax.transAxes.inverted().transform((0,label_top))[1]
        ax.xaxis.label.set_va('top');ax.xaxis.set_label_coords(.5,y)
    if not is_comp:
        # Keep original annotation contents and vertical positions; prevent
        # annotation text from extending into the y-axis title/tick labels.
        fig.canvas.draw();ren=fig.canvas.get_renderer();ab=ax.get_window_extent(ren)
        pad=1.2/25.4*fig.dpi
        breakpoint_x = None
        for line in ax.lines:
            x_values = np.asarray(line.get_xdata(), dtype=float)
            if line.get_linestyle() == '--' and len(x_values) == 2 and x_values[0] == x_values[1]:
                breakpoint_x = ax.transData.transform((x_values[0], 0))[0]
                break
        for t in ax.texts:
            box=t.get_window_extent(ren)
            right = ab.x1-pad
            if t is ax.texts[0] and breakpoint_x is not None:
                # Keep the lower-stage annotation clear of the breakpoint.
                right = min(right, breakpoint_x-0.5/25.4*fig.dpi)
            dx=max(0,ab.x0+pad-box.x0)-max(0,box.x1-right)
            if dx:
                px,py=t.get_transform().transform(t.get_position())
                t.set_position(t.get_transform().inverted().transform((px+dx,py)))
    # Panel letters are included at a fixed page coordinate, outside the axes.
    panel_letter_size = fonts.get("panel_letter", LAYOUT["panel_letter"]["font_pt"])
    panel_letter(fig,name[-1].upper(),width,height,font_pt=panel_letter_size)
    assert plot_signature(fig)==before,name
    fig.canvas.draw();ren=fig.canvas.get_renderer()
    axbox=ax.get_window_extent(ren)
    texts=[*ax.texts,ax.xaxis.label,ax.yaxis.label,ax.title,
           *ax.get_xticklabels(),*ax.get_yticklabels(),*fig.texts]
    boxes=[]
    for t in texts:
        if not t.get_visible() or not t.get_text():continue
        b=t.get_window_extent(ren)
        assert b.x0>=-.5 and b.y0>=-.5 and b.x1<=fig.bbox.width+.5 and b.y1<=fig.bbox.height+.5,(name,t.get_text(),b.bounds,fig.bbox.bounds)
        boxes.append(dict(text=t.get_text(),display_text=t.get_text().replace(r'$\beta$', 'β'),font_pt=t.get_fontsize(),
                          rotation=t.get_rotation(),bounds_mm=(np.array(b.bounds)/fig.dpi*25.4).tolist()))
    with plt.rc_context({'savefig.bbox':None}):
        fig.savefig(FIG/f'{name}.pdf',dpi=400,bbox_inches=None)
        fig.savefig(FIG/f'{name}.png',dpi=LAYOUT['png_dpi'],bbox_inches=None)
    record=dict(canvas_mm=[width,height],axes_rect_mm=[left,bottom,aw,ah],
        axes_measured_mm=(np.array(axbox.bounds)/fig.dpi*25.4).tolist(),
        xlim=list(ax.get_xlim()),ylim=list(ax.get_ylim()),
        original_graphics_scale=scale,plotted_arrays_colors_alpha_unchanged=True,
        visible_method_label_changes=visible_label_changes,
        fonts_pt={'tick':tick_size,'annotation':annotation_size,'x_axis_title':7.5 if is_comp else axis_title_size,'y_axis_title':axis_title_size,'panel_title':panel_title_size,'panel_letter':panel_letter_size},
        text_bounds=boxes)
    plt.close(fig)
    return record
