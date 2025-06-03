# custom_legend_handler_v2.py

import matplotlib
from matplotlib.legend_handler import HandlerBase
from matplotlib.text import Text
from matplotlib.lines import Line2D
import numpy as np  # 用于 np.array, np.round


class HandlerInterspersedText(HandlerBase):
    """
    图例处理器，用于显示：[横线] Text1 [标记] Text2.
    允许分别控制元素间距。
    """

    def __init__(self, text1_content, text2_content,
                 default_xpad_points=1.5,  # 常规元素间间距 (点)
                 pad_after_text1_points=15.0,  # “预计”文本后的特定间距 (点)
                 text_color=None,
                 **kwargs):
        super().__init__(**kwargs)
        self._text1 = text1_content
        self._text2 = text2_content
        self.default_xpad_points = default_xpad_points
        self.pad_after_text1_points = pad_after_text1_points
        self.custom_text_color = text_color

    def create_artists(self, legend, orig_handle,
                       xdescent, ydescent, width, height, fontsize,  # width, height, fontsize 都是点单位
                       trans):  # 这个 trans 是 HandleBox 的变换

        if not (isinstance(orig_handle, tuple) and len(orig_handle) == 2):
            print("警告: HandlerInterspersedText 接收到非预期的 orig_handle 格式。")
            return super().create_artists(legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans)

        line_proxy = orig_handle[0]
        marker_proxy = orig_handle[1]

        resolved_text_color = 'black'
        if self.custom_text_color:
            resolved_text_color = self.custom_text_color
        elif line_proxy:
            resolved_text_color = line_proxy.get_color()

        # DPI 获取 (主要用于精确文本宽度计算时的单位转换，如果renderer可用)
        dpi = matplotlib.rcParams.get('figure.dpi', 96.0)  # 默认DPI
        try:
            if legend.figure and legend.figure.canvas and hasattr(legend.figure.canvas, 'get_renderer'):
                renderer = legend.figure.canvas.get_renderer()
                if renderer and hasattr(renderer, 'dpi') and renderer.dpi != 0:
                    dpi = renderer.dpi
        except AttributeError:
            print("DEBUG: 获取 renderer.dpi 时可能出错，继续使用 rcParams['figure.dpi']。")

        # --- 根据 HandleBox 高度调整内部字体大小 (单位：点) ---
        font_height_allowance_factor = 0.80
        pixels_to_points = 72.0 / dpi  # 1 pixel = 72/dpi points
        # height (传入的是点单位) * font_height_allowance_factor
        max_font_height_pt = height * font_height_allowance_factor

        desired_scale_factor = 0.70
        desired_internal_fontsize_pt = fontsize * desired_scale_factor

        internal_fontsize_pt = min(desired_internal_fontsize_pt, max_font_height_pt)
        internal_fontsize_pt = max(internal_fontsize_pt, 4.0)  # 最小4点
        # --- 字体大小调整结束 ---

        # 其他尺寸因子
        line_length_factor = 0.18
        text_width_estimate_factor = 0.55  # 文本宽度估算因子 (相对于 internal_fontsize_pt)
        marker_spacing_scale_factor = 0.45
        internal_linewidth_factor = 0.70
        internal_markersize_factor = 0.75

        artists = []
        current_x_position_pt = 0.0  # 当前绘制位置的 X 坐标 (点单位)

        # 1. 绘制横线
        line_length_pt = legend.handlelength * fontsize * line_length_factor
        if line_proxy:
            line = Line2D([current_x_position_pt, current_x_position_pt + line_length_pt],
                          [height / 2.0, height / 2.0],
                          linestyle=line_proxy.get_linestyle(),
                          color=line_proxy.get_color(),
                          linewidth=max(0.5, line_proxy.get_linewidth() * internal_linewidth_factor),
                          transform=trans)
            artists.append(line)
        current_x_position_pt += line_length_pt + self.default_xpad_points

        # 2. 绘制第一个文本 ("预计")
        if self._text1:
            text1_artist = Text(x=current_x_position_pt, y=height / 2.0,
                                text=self._text1, color=resolved_text_color,
                                fontproperties=legend.prop, fontsize=internal_fontsize_pt,
                                ha='left', va='center', transform=trans)
            artists.append(text1_artist)

            text1_width_pt = 0;
            renderer = None
            try:
                if legend.figure.canvas and hasattr(legend.figure.canvas, 'get_renderer'):
                    renderer = legend.figure.canvas.get_renderer()
                if renderer:
                    bbox_px = text1_artist.get_window_extent(renderer=renderer)
                    text1_width_pt = bbox_px.width * pixels_to_points  # 像素转回点
                else:
                    text1_width_pt = len(self._text1) * internal_fontsize_pt * text_width_estimate_factor
            except Exception as e_text_width:
                print(f"DEBUG: 获取文本1宽度时出错({type(e_text_width).__name__}): {e_text_width}, 使用估算值。")
                text1_width_pt = len(self._text1) * internal_fontsize_pt * text_width_estimate_factor

            current_x_position_pt += text1_width_pt + self.pad_after_text1_points  # 使用“预计”后的特定间距

        # 3. 绘制标记 (点)
        marker_footprint_pt = 0
        if marker_proxy:
            actual_marker_draw_size_pt = marker_proxy.get_markersize() * legend.markerscale * internal_markersize_factor
            effective_marker_size_for_spacing_pt = actual_marker_draw_size_pt * marker_spacing_scale_factor
            marker_footprint_pt = effective_marker_size_for_spacing_pt

            marker_artist = Line2D([current_x_position_pt], [height / 2.0],
                                   marker=marker_proxy.get_marker(), markersize=actual_marker_draw_size_pt,
                                   color=marker_proxy.get_color(), markerfacecolor=marker_proxy.get_markerfacecolor(),
                                   markeredgecolor=marker_proxy.get_markeredgecolor(),
                                   linestyle='None', transform=trans)
            artists.append(marker_artist)
            current_x_position_pt += marker_footprint_pt + self.default_xpad_points
        elif self._text2:
            current_x_position_pt += self.default_xpad_points

            # 4. 绘制第二个文本 (例如 "实际 (时间t)")
        if self._text2:
            text2_artist = Text(x=current_x_position_pt, y=height / 2.0,
                                text=self._text2, color=resolved_text_color,
                                fontproperties=legend.prop, fontsize=internal_fontsize_pt,
                                ha='left', va='center', transform=trans)
            artists.append(text2_artist)
            text2_width_pt = 0;
            renderer = None
            try:
                if legend.figure.canvas and hasattr(legend.figure.canvas, 'get_renderer'):
                    renderer = legend.figure.canvas.get_renderer()
                if renderer:
                    bbox_px = text2_artist.get_window_extent(renderer=renderer)
                    text2_width_pt = bbox_px.width * pixels_to_points  # 像素转回点
                else:
                    text2_width_pt = len(self._text2) * internal_fontsize_pt * text_width_estimate_factor
            except Exception as e_text_width:
                print(f"DEBUG: 获取文本2宽度时出错({type(e_text_width).__name__}): {e_text_width}, 使用估算值。")
                text2_width_pt = len(self._text2) * internal_fontsize_pt * text_width_estimate_factor
            current_x_position_pt += text2_width_pt

        print(f"DEBUG Handler: T1='{self._text1}', T2='{self._text2}', Artists: {len(artists)}")
        print(
            f"  HandleBox(pts): w={width:.2f}, h={height:.2f}, legend_fs={fontsize:.2f}, internal_fs_pt={internal_fontsize_pt:.2f}")
        print(f"  Paddings(pts): default={self.default_xpad_points}, after_text1={self.pad_after_text1_points}")
        # 详细打印每个艺术家的属性 (可选，但对调试非常有用)
        for i, art in enumerate(artists):
            print(f"  Artist {i}: {type(art)}")
            if isinstance(art, Text):
                print(
                    f"    Text: '{art.get_text()}', x_pt: {art.get_position()[0]:.2f}, y_pt: {art.get_position()[1]:.2f}, fs_pt: {art.get_fontsize()}")
            elif isinstance(art, Line2D):
                print(
                    f"    Line: xdata_pt: {np.array(art.get_xdata()).round(2)}, ydata_pt: {np.array(art.get_ydata()).round(2)}, lw_pt: {art.get_linewidth()}, marker: {art.get_marker()}, ms_pt: {art.get_markersize()}")
        print(f"  Calculated total width (points): {current_x_position_pt:.2f} (vs HandleBox width_pt: {width:.2f})")

        return artists