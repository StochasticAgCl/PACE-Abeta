# Layout-only publication rendering of the memory analysis plots.
# No model is fitted and no source data, source script, or upstream artifact is edited.
# The original script loads the saved model and evaluates its existing predictions.
args <- commandArgs(trailingOnly=TRUE)
if (length(args) != 1) stop("Usage: figure4.R WORK_DIR")
root <- normalizePath(args[1])
for (folder in c("figures/original_style", "data/analysis", "reports/analysis", "tmp"))
  dir.create(file.path(root,folder),recursive=TRUE,showWarnings=FALSE)
script_arg <- grep("^--file=", commandArgs(trailingOnly=FALSE), value=TRUE)
script_dir <- dirname(normalizePath(sub("^--file=", "", script_arg[1])))
layout <- jsonlite::fromJSON(file.path(script_dir,"figure_layout.json"))
PNG_DPI <- layout$png_dpi
fonts <- layout$figure3$fonts_pt
source_path <- file.path(script_dir,"memory_plots.R")
input_dir <- file.path(root,"data/analysis/memory")
protected <- c(source_path, file.path(input_dir,c("dat_lmm.csv","matched_subjects.csv","wald_slope_results.csv","comparative_model_rma_mv.rds")))
hashes_before <- tools::md5sum(protected)
source(source_path)
library(grid)
library(gtable)

originals <- list(A=p_balance,B=p_trend_adni)
original_builds <- lapply(originals,ggplot_build)
plots <- lapply(originals, function(p) {
  p <- unserialize(serialize(p,NULL))
  # Font sizes refer to the standalone panel; manuscript scaling is uniform.
  p <- p + theme(
    axis.title = element_text(size=fonts$axis_title,color="black"),
    axis.text = element_text(size=fonts$tick,color="black"),
    axis.title.y = element_text(margin=margin(r=0.5)),
    axis.text.y = element_text(margin=margin(r=1)),
    plot.title = element_text(size=fonts$panel_title,color="black",hjust=0.5,face="plain"),
    plot.subtitle = element_text(size=fonts$axis_title,color="black",hjust=0.5),
    legend.title = element_text(size=fonts$legend_title,color="black"),
    legend.text = element_text(size=fonts$legend_text,color="black"),
    legend.position = "bottom",
    legend.direction = "horizontal",
    legend.box = "horizontal",
    legend.justification = "center",
    legend.key.width = unit(4,"mm"),
    legend.key.height = unit(3.2,"mm"),
    legend.margin = margin(0,0,0,0,"mm"),
    legend.box.margin = margin(0,0,0,0,"mm"),
    legend.box.spacing = unit(0.8,"mm"),
    legend.spacing.x = unit(1,"mm"),
    plot.margin = margin(1.5,0.8,1.2,0,"mm")
  ) + guides(
    fill = guide_legend(nrow=1,title.position="left",byrow=TRUE),
    color = guide_legend(nrow=1,title.position="left",byrow=TRUE)
  )
  for(i in seq_along(p$layers)) {
    if(inherits(p$layers[[i]]$geom,"GeomText")) {
      p$layers[[i]]$aes_params$size <- fonts$annotation/ggplot2::.pt
      p$layers[[i]]$aes_params$fontface <- "plain"
    }
  }
  p
})
# Display-only y headroom and fixed count ticks leave counts unchanged,
# so the original upper-right p value clears the highest histogram bar.
plots$A <- plots$A + scale_y_continuous(breaks=c(0,2,4,6,8),
                                      expand=expansion(mult=c(0.05,0.23)))

# Verify data, scales, every statistical layer, original annotation contents and
# positions, palette, line widths, and transparency. Only annotation font size
# and font face are omitted: these are typography-only changes.
strip_annotation_typography <- function(b,p) {
  for(i in seq_along(p$layers)) if(inherits(p$layers[[i]]$geom,"GeomText")) {
    b$data[[i]]$size <- NULL
    b$data[[i]]$fontface <- NULL
  }
  list(data=b$data,panel=lapply(b$layout$panel_params,function(x) {
    list(x_range=x$x.range,y_range=x$y.range,
         x_breaks=x$x$breaks,y_breaks=x$y$breaks)
  }))
}
for(i in seq_along(plots)) {
  before <- strip_annotation_typography(original_builds[[i]],originals[[i]])
  after <- strip_annotation_typography(ggplot_build(plots[[i]]),plots[[i]])
  if(i==1) {
    # Matching-panel headroom and fixed display ticks may differ from the
    # original automatic ticks. ggplot2 marks out-of-range ticks as NA.
    final_breaks <- plots[[i]]$scales$get_scales("y")$breaks
    stopifnot(identical(final_breaks,c(0,2,4,6,8)))
    expected_breaks <- final_breaks
    final_range <- after$panel[[1]]$y_range
    expected_breaks[expected_breaks < final_range[1] |
                    expected_breaks > final_range[2]] <- NA_real_
    stopifnot(identical(before$panel[[1]]$x_range,after$panel[[1]]$x_range),
              identical(before$panel[[1]]$x_breaks,after$panel[[1]]$x_breaks),
              identical(after$panel[[1]]$y_breaks,expected_breaks))
    before$panel[[1]]$y_range <- NULL; after$panel[[1]]$y_range <- NULL
    before$panel[[1]]$y_breaks <- NULL; after$panel[[1]]$y_breaks <- NULL
  }
  stopifnot(identical(before,after))
}

SLOT <- layout$figure3$canvas_mm[1]
HEIGHT <- layout$figure3$canvas_mm[2]
axis_rect <- layout$figure3$axes_rect_mm
AXIS_X <- axis_rect[1]; AXIS_Y <- axis_rect[2]
PANEL_W <- axis_rect[3]; PANEL_H <- axis_rect[4]
# Measure on the rendering device, then anchor the coordinate rectangle to the
# same physical origin used by the Python panels.
cairo_pdf(file.path(root,"tmp/memory_geometry_measure.pdf"),width=SLOT/25.4,height=HEIGHT/25.4)
grobs <- lapply(plots,ggplotGrob)
common_w <- do.call(unit.pmax,lapply(grobs,function(g) g$widths))
common_h <- do.call(unit.pmax,lapply(grobs,function(g) g$heights))
for(i in seq_along(grobs)) {
  g <- grobs[[i]]
  ip <- which(g$layout$name=="panel")
  stopifnot(length(ip)==1)
  g$widths <- common_w; g$heights <- common_h
  g$widths[g$layout$l[ip]] <- unit(PANEL_W,"mm")
  g$heights[g$layout$t[ip]] <- unit(PANEL_H,"mm")
  g$widths <- unit(convertWidth(g$widths,"mm",TRUE),"mm")
  g$heights <- unit(convertHeight(g$heights,"mm",TRUE),"mm")
  grobs[[i]] <- g
}
gw <- convertWidth(sum(grobs[[1]]$widths),"mm",TRUE)
gh <- convertHeight(sum(grobs[[1]]$heights),"mm",TRUE)
ip <- which(grobs[[1]]$layout$name=="panel")
col <- grobs[[1]]$layout$l[ip]; row <- grobs[[1]]$layout$t[ip]
left_internal <- sum(convertWidth(grobs[[1]]$widths[seq_len(col-1)],"mm",TRUE))
top_internal <- sum(convertHeight(grobs[[1]]$heights[seq_len(row-1)],"mm",TRUE))
bottom_internal <- gh-top_internal-PANEL_H
left_offset <- AXIS_X-left_internal
bottom_offset <- AXIS_Y-bottom_internal
stopifnot(left_offset>=0,bottom_offset>=0,
          left_offset+gw<=SLOT,bottom_offset+gh<=HEIGHT)
measured_rect <- c(left_offset+left_internal,bottom_offset+bottom_internal,
                  convertWidth(grobs[[1]]$widths[col],"mm",TRUE),
                  convertHeight(grobs[[1]]$heights[row],"mm",TRUE))
stopifnot(isTRUE(all.equal(measured_rect,axis_rect,tolerance=1e-12)))
for(g in grobs) stopifnot(identical(g$widths,grobs[[1]]$widths),identical(g$heights,grobs[[1]]$heights))
grob_rect_mm <- function(g,index) {
  cell <- g$layout[index,]
  widths <- convertWidth(g$widths,"mm",TRUE)
  heights <- convertHeight(g$heights,"mm",TRUE)
  c(left_offset+sum(widths[seq_len(cell$l-1)]),
    bottom_offset+gh-sum(heights[seq_len(cell$b)]),
    sum(widths[cell$l:cell$r]),sum(heights[cell$t:cell$b]))
}
legend_geometry <- lapply(grobs,function(g) {
  legend_index <- which(g$layout$name=="guide-box-bottom")
  xlabel_index <- which(g$layout$name=="xlab-b")
  stopifnot(length(legend_index)==1,length(xlabel_index)==1)
  legend_rect <- grob_rect_mm(g,legend_index)
  xlabel_rect <- grob_rect_mm(g,xlabel_index)
  stopifnot(legend_rect[2]+legend_rect[4]<=xlabel_rect[2]+1e-9)
  legend <- g$grobs[[legend_index]]
  guides <- legend$grobs[legend$layout$name=="guides"]
  stopifnot(length(guides)==1)
  key_rows <- guides[[1]]$layout$t[grepl("^key-",guides[[1]]$layout$name)]
  stopifnot(length(unique(key_rows))==1)
  list(legend_rect_mm=legend_rect,x_axis_label_rect_mm=xlabel_rect,
       rows=1,below_x_axis_label=TRUE)
})
dev.off()

render <- function(path,i) {
  if(grepl("\\.pdf$",path)) cairo_pdf(path,width=SLOT/25.4,height=HEIGHT/25.4,onefile=FALSE)
  else png(path,width=floor(SLOT*PNG_DPI/25.4),height=floor(HEIGHT*PNG_DPI/25.4),
           units="px",res=PNG_DPI,type="cairo",bg="white")
  grid.newpage()
  pushViewport(viewport(x=unit(left_offset+gw/2,"mm"),y=unit(bottom_offset+gh/2,"mm"),
                        width=unit(gw,"mm"),height=unit(gh,"mm")))
  grid.draw(grobs[[i]])
  popViewport()
  dev.off()
}
panel_names <- c("figure3c","figure3d")
for(i in seq_along(grobs)) for(ext in c("pdf","png"))
  render(file.path(root,"figures",paste0(panel_names[i],".",ext)),i)
# Cairo rounds PDF page boxes to whole points. Its physical drawing coordinates
# remain anchored at the bottom left; restoring the boxes needs no translation.
box_script <- tempfile(fileext=".py")
writeLines(c(
  "import sys, os",
  "from pypdf import PdfReader, PdfWriter",
  "width,height=map(float,sys.argv[1:3])",
  "for path in sys.argv[3:]:",
  "    reader=PdfReader(path); writer=PdfWriter()",
  "    for page in reader.pages:",
  "        page.mediabox.upper_right=(width*72/25.4,height*72/25.4)",
  "        page.cropbox.upper_right=(width*72/25.4,height*72/25.4)",
  "        writer.add_page(page)",
  "    tmp=path+'.pagebox.tmp'",
  "    with open(tmp,'wb') as f: writer.write(f)",
  "    os.replace(tmp,path)"
),box_script)
box_status <- system2(Sys.getenv("PACE_PYTHON", "python3"),
  c(shQuote(box_script),as.character(SLOT),as.character(HEIGHT),
    shQuote(file.path(root,"figures",paste0(panel_names,".pdf")))))
unlink(box_script)
stopifnot(box_status==0)

stopifnot(identical(hashes_before,tools::md5sum(protected)))
audit <- list(
  source=source_path,upstream_inputs_unchanged=TRUE,protected_input_md5=as.list(hashes_before),
  fit_reestimated=FALSE,matched_sample_reselected=FALSE,
  original_statistical_layers_annotation_content_positions_colors_alpha_linewidths_unchanged=TRUE,
  original_scale_preserved_except_matching_y_range_and_ticks=TRUE,
  matching_y_range_original=original_builds$A$layout$panel_params[[1]]$y.range,
  matching_y_range_final=ggplot_build(plots$A)$layout$panel_params[[1]]$y.range,
  printed_size_mm=list(single=c(SLOT,HEIGHT)),
  panels=setNames(lapply(seq_along(grobs),function(i) list(
    canvas_mm=c(SLOT,HEIGHT),axes_rect_mm=measured_rect,
    axes_measured_mm=measured_rect,
    legend=legend_geometry[[i]],
    plotted_arrays_colors_alpha_unchanged=TRUE,
    fonts_pt=list(tick=fonts$tick,annotation=fonts$annotation,
                  x_axis_title=fonts$axis_title,y_axis_title=fonts$axis_title,
                  panel_title=fonts$panel_title,legend_text=fonts$legend_text,
                  legend_title=fonts$legend_title)
  )),panel_names),
  same_coordinate_rectangle=TRUE,
  fonts_panel_pt=fonts,
  annotation_face="plain",font_family="original sans",legend_position="bottom",
  legend_direction="horizontal",legend_rows=1,
  presentation_changes=list(
    "Coordinate rectangle, canvas size, and placement use the shared figure layout.",
    "Identical physical layout rows and columns align both memory panels.",
    "Font hierarchy is read from the shared Figure 3 configuration.",
    "Single-row legends sit below the x-axis labels, with 4 mm keys and 0.8 mm spacing.",
    "Outer plot margins: 1.5 mm top, 1.2 mm bottom, 0 mm left, and 0.8 mm right.",
    "Matching y-range gains upper headroom to clear the original upper-right p-value annotation; fixed count ticks are used. Histogram and density statistics are unchanged."
  )
)
jsonlite::write_json(audit,file.path(root,"reports/analysis/memory_figure_geometry.json"),auto_unbox=TRUE,pretty=TRUE,digits=16)
print(audit$panels)
cat("Rendered Figure 3C/D with the shared physical coordinate rectangles.\n")
