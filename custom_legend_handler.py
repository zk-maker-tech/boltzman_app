# custom_legend_handler.py

import matplotlib  # 仍然需要它，以防万一（例如 rcParams 的某些默认值）
from matplotlib.legend_handler import HandlerBase
from matplotlib.text import Text
from matplotlib.lines import Line2D
import numpy as np  # 用于 np.array, np.round


class HandlerInterspersedText(HandlerBase):
    def __init__(self, text1_content, text2_content,
                 xpad_points=1.0, text_color=None,
                 **kwargs):
        super().__init__(**kwargs)
        self._text1 = text1_content
        self._text2 = text2_content
        self.xpad_points = xpad_points  # 明确这是点单位的间距
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

        # --- 根据 HandleBox 高度调整内部字体大小 (单位：点) ---
        font_height_allowance_factor = 0.80
        max_font_height_pt = height * font_height_allowance_factor

        desired_scale_factor = 0.70
        desired_internal_fontsize_pt = fontsize * desired_scale_factor

        internal_fontsize_pt = min(desired_internal_fontsize_pt, max_font_height_pt)
        internal_fontsize_pt = max(internal_fontsize_pt, 4.0)  # 最小4点
        # --- 字体大小调整结束 ---

        # 其他尺寸因子 (基于调整后的 internal_fontsize_pt 或原始 fontsize)
        line_length_factor = 0.20  # 线条长度因子 (相对于 legend.handlelength * fontsize)
        text_width_estimate_factor = 0.60  # 文本宽度估算因子 (相对于 internal_fontsize_pt), 稍微调大一点
        marker_spacing_scale_factor = 0.5  # 标记占位宽度因子 (相对于其绘制尺寸)
        internal_linewidth_factor = 0.8
        internal_markersize_factor = 0.8

        artists = []
        current_x_position_pt = 0.0  # 当前绘制位置的 X 坐标 (点单位)

        # 1. 绘制横线
        # 线条长度基于 legend.handlelength * (图例原始fontsize) * 因子 (单位：点)
        line_length_pt = legend.handlelength * fontsize * line_length_factor

        if line_proxy:
            line = Line2D([current_x_position_pt, current_x_position_pt + line_length_pt],  # x坐标 (点)
                          [height / 2.0, height / 2.0],  # y坐标 (点, 垂直居中)
                          linestyle=line_proxy.get_linestyle(),
                          color=line_proxy.get_color(),
                          linewidth=max(0.5, line_proxy.get_linewidth() * internal_linewidth_factor),  # 线宽 (点)
                          transform=trans)  # << 使用传入的变换
            artists.append(line)
        current_x_position_pt += line_length_pt + self.xpad_points  # xpad_points 已是点单位

        # 2. 绘制第一个文本 ("预计")
        if self._text1:
            text1_artist = Text(x=current_x_position_pt, y=height / 2.0,  # x,y 坐标 (点)
                                text=self._text1,
                                color=resolved_text_color,
                                fontproperties=legend.prop,  # legend.prop 会处理字体
                                fontsize=internal_fontsize_pt,  # 字体大小 (点)
                                ha='left',
                                va='center',
                                transform=trans)  # << 使用传入的变换
            artists.append(text1_artist)

            # 估算文本1的宽度 (点单位)
            text1_width_pt = 0
            renderer = None
            try:
                # 尝试获取renderer，如果成功，精确计算宽度并转为点
                if legend.figure.canvas and hasattr(legend.figure.canvas, 'get_renderer'):
                    renderer = legend.figure.canvas.get_renderer()
                if renderer:
                    bbox_px = text1_artist.get_window_extent(renderer=renderer)
                    # get_window_extent 返回的是像素，我们需要转换回点单位进行布局
                    dpi = renderer.dpi if renderer.dpi != 0 else matplotlib.rcParams.get('figure.dpi', 96.0)
                    pixels_to_points = 72.0 / dpi
                    text1_width_pt = bbox_px.width * pixels_to_points
                else:  # Renderer 不可用，使用估算
                    # print("DEBUG: Text1 renderer 为 None, 使用估算宽度。")
                    text1_width_pt = len(self._text1) * internal_fontsize_pt * text_width_estimate_factor
            except Exception as e_text_width:
                print(f"DEBUG: 获取文本1宽度时出错({type(e_text_width).__name__}): {e_text_width}, 使用估算值。")
                text1_width_pt = len(self._text1) * internal_fontsize_pt * text_width_estimate_factor

            current_x_position_pt += text1_width_pt + self.xpad_points

        # 3. 绘制标记 (点)
        marker_footprint_pt = 0
        if marker_proxy:
            # markersize 单位是点, legend.markerscale 是图例全局缩放因子
            actual_marker_draw_size_pt = marker_proxy.get_markersize() * legend.markerscale * internal_markersize_factor

            effective_marker_size_for_spacing_pt = actual_marker_draw_size_pt * marker_spacing_scale_factor
            marker_footprint_pt = effective_marker_size_for_spacing_pt

            marker_artist = Line2D([current_x_position_pt], [height / 2.0],  # x,y 坐标 (点)
                                   marker=marker_proxy.get_marker(),
                                   markersize=actual_marker_draw_size_pt,  # 标记大小 (点)
                                   color=marker_proxy.get_color(),
                                   markerfacecolor=marker_proxy.get_markerfacecolor(),
                                   markeredgecolor=marker_proxy.get_markeredgecolor(),
                                   linestyle='None',
                                   transform=trans)  # << 使用传入的变换
            artists.append(marker_artist)
            current_x_position_pt += marker_footprint_pt + self.xpad_points
        elif self._text2:
            current_x_position_pt += self.xpad_points

        # 4. 绘制第二个文本 (例如 "实际 (时间t)")
        if self._text2:
            text2_artist = Text(x=current_x_position_pt, y=height / 2.0,  # x,y 坐标 (点)
                                text=self._text2,
                                color=resolved_text_color,
                                fontproperties=legend.prop,
                                fontsize=internal_fontsize_pt,  # 字体大小 (点)
                                ha='left',
                                va='center',
                                transform=trans)  # << 使用传入的变换
            artists.append(text2_artist)
            text2_width_pt = 0
            renderer = None
            try:
                if legend.figure.canvas and hasattr(legend.figure.canvas, 'get_renderer'):
                    renderer = legend.figure.canvas.get_renderer()
                if renderer:
                    bbox_px = text2_artist.get_window_extent(renderer=renderer)
                    dpi = renderer.dpi if renderer.dpi != 0 else matplotlib.rcParams.get('figure.dpi', 96.0)
                    pixels_to_points = 72.0 / dpi
                    text2_width_pt = bbox_px.width * pixels_to_points
                else:
                    # print("DEBUG: Text2 renderer 为 None, 使用估算宽度。")
                    text2_width_pt = len(self._text2) * internal_fontsize_pt * text_width_estimate_factor
            except Exception as e_text_width:
                print(f"DEBUG: 获取文本2宽度时出错({type(e_text_width).__name__}): {e_text_width}, 使用估算值。")
                text2_width_pt = len(self._text2) * internal_fontsize_pt * text_width_estimate_factor
            current_x_position_pt += text2_width_pt

        print(
            f"DEBUG: Handler (Full Point-Based): Text1='{self._text1}', Text2='{self._text2}', Artists created: {len(artists)}")
        print(
            f"  DEBUG: HandleBox dimensions (points): pt_w={width:.2f}, pt_h={height:.2f}, legend_fontsize={fontsize:.2f}, ADAPTED internal_fontsize_pt={internal_fontsize_pt:.2f}")
        for i, art in enumerate(artists):
            print(f"  DEBUG: Artist {i}: {type(art)}")
            if isinstance(art, Text):
                # Text.get_position() 返回的是相对于其变换原点的坐标
                print(
                    f"    Text content: '{art.get_text()}', set_x_pt: {art.get_position()[0]:.2f}, set_y_pt: {art.get_position()[1]:.2f}, color: {art.get_color()}, fontsize: {art.get_fontsize()}")
            elif isinstance(art, Line2D):
                x_data_arr = np.array(art.get_xdata())  # 这些已经是我们设置的点单位坐标
                y_data_arr = np.array(art.get_ydata())
                print(
                    f"    Line2D color: {art.get_color()}, marker: {art.get_marker()}, linestyle: {art.get_linestyle()}, linewidth: {art.get_linewidth()}")
                print(f"      xdata_pt: {x_data_arr.round(2)}, ydata_pt: {y_data_arr.round(2)}")
                if art.get_marker() != 'None' and art.get_marker() is not None:
                    print(f"      markersize_pt: {art.get_markersize()}")
        print(
            f"  DEBUG: Calculated total width for elements (points): {current_x_position_pt:.2f} (compare with HandleBox width_pt: {width:.2f})")

        return artists