# src/algorithms/dcp.py
import cv2
import numpy as np


class DCPDehazer:
    """
    Dark Channel Prior (DCP) 图像去雾算法实现.

    约定:
    - 输入图像: uint8, BGR, shape (H, W, 3)
    - 内部计算: float32, [0,1]
    - 输出图像: uint8, BGR, shape (H, W, 3)
    """

    def __init__(
        self,
        patch_size: int = 15,
        omega: float = 0.95,
        t0: float = 0.1,
        use_guided_filter: bool = True,
        guided_radius: int = 40,
        guided_eps: float = 1e-3,
    ):
        """
        :param patch_size: 暗通道最小滤波窗口大小 (必须为奇数, 一般 15)
        :param omega: 缩放系数, 通常 0.95
        :param t0: 透射率下限, 防止除 0, 通常 0.1
        :param use_guided_filter: 是否使用引导滤波细化透射率
        :param guided_radius: 引导滤波半径
        :param guided_eps: 引导滤波正则化项
        """
        if patch_size % 2 == 0:
            raise ValueError("patch_size must be odd.")
        self.patch_size = patch_size
        self.omega = omega
        self.t0 = t0
        self.use_guided_filter = use_guided_filter
        self.guided_radius = guided_radius
        self.guided_eps = guided_eps

    # ---------------------- 工具函数 ---------------------- #

    @staticmethod
    def _to_float(img: np.ndarray) -> np.ndarray:
        """uint8 -> float32, [0,1]"""
        if img is None:
            raise ValueError("Input image is None.")
        if img.dtype == np.uint8:
            return img.astype(np.float32) / 255.0
        return img.astype(np.float32)

    @staticmethod
    def _to_uint8(img: np.ndarray) -> np.ndarray:
        """float32 [0,1] -> uint8 [0,255]"""
        img = np.clip(img * 255.0, 0, 255)
        return img.astype(np.uint8)

    # ---------------------- 核心步骤 ---------------------- #

    def dark_channel(self, img: np.ndarray) -> np.ndarray:
        """
        计算暗通道:
        J_dark(x) = min_{y∈Ω(x)}( min_c I_c(y) )

        :param img: float32, [0,1], (H,W,3)
        :return: 暗通道图 (H,W)
        """
        # 先对通道上取 min
        min_per_pixel = np.min(img, axis=2)
        # 再用最小滤波（形态学腐蚀）在局部窗口内取 min
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (self.patch_size, self.patch_size)
        )
        dark = cv2.erode(min_per_pixel, kernel)
        return dark

    def estimate_atmospheric_light(
        self, img: np.ndarray, dark: np.ndarray
    ) -> np.ndarray:
        """
        根据暗通道估计大气光 A.

        做法: 取暗通道中最亮的前 0.1% 像素，对应到原图，在其中选亮度最大的像素作为 A.

        :param img: float32, [0,1], (H,W,3)
        :param dark: 暗通道 (H,W)
        :return: A, shape (3,)
        """
        h, w = dark.shape
        n_pixels = h * w
        n_search = max(int(n_pixels * 0.001), 1)  # top 0.1%

        dark_vec = dark.reshape(-1)
        img_vec = img.reshape(-1, 3)

        # 暗通道值从小到大排序，取最亮的 n_search 个
        indices = np.argsort(dark_vec)[-n_search:]
        candidates = img_vec[indices]

        # 在这些候选中，取亮度 (R+G+B) 最大的一个像素作为 A
        intensity = np.sum(candidates, axis=1)
        A = candidates[np.argmax(intensity)]
        return A  # (3,)

    def estimate_transmission(self, img: np.ndarray, A: np.ndarray) -> np.ndarray:
        """
        估计粗略透射率 t(x):

        t(x) = 1 - ω * min_{y∈Ω(x)} ( min_c I_c(y) / A_c )

        :param img: float32, [0,1], (H,W,3)
        :param A: 大气光, shape (3,)
        :return: 粗略透射率 t, (H,W)
        """
        normed = img / A.reshape(1, 1, 3)
        normed_dark = self.dark_channel(normed)
        t = 1.0 - self.omega * normed_dark
        return t

    def refine_transmission(self, img: np.ndarray, t: np.ndarray) -> np.ndarray:
        """
        使用引导滤波细化透射率.
        若无 ximgproc 模块, 回退到简单的均值模糊.

        :param img: float32, [0,1], (H,W,3)
        :param t: 粗略透射率 (H,W)
        :return: 细化后的透射率 (H,W)
        """
        if not self.use_guided_filter:
            return t

        # 用灰度图作为引导图
        gray = cv2.cvtColor(self._to_uint8(img), cv2.COLOR_BGR2GRAY).astype(
            np.float32
        ) / 255.0

        try:
            # opencv-contrib-python 提供的 guidedFilter
            guided = cv2.ximgproc.guidedFilter(
                guide=gray,
                src=t.astype(np.float32),
                radius=self.guided_radius,
                eps=self.guided_eps,
            )
            return guided
        except Exception:
            # 没装 ximgproc 时，使用简单模糊代替
            ksize = max(3, self.guided_radius // 2 * 2 + 1)  # 确保为奇数
            return cv2.blur(t.astype(np.float32), (ksize, ksize))

    def recover(self, img: np.ndarray, t: np.ndarray, A: np.ndarray) -> np.ndarray:
        """
        根据成像模型恢复无雾图像:

        J(x) = (I(x) - A) / max(t(x), t0) + A

        :param img: float32, [0,1], (H,W,3)
        :param t: 透射率 (H,W)
        :param A: 大气光 (3,)
        :return: J, float32, [0,1], (H,W,3)
        """
        t = np.clip(t, self.t0, 1.0)
        t = t[:, :, np.newaxis]  # (H,W,1)

        J = (img - A.reshape(1, 1, 3)) / t + A.reshape(1, 1, 3)
        J = np.clip(J, 0.0, 1.0)
        return J

    # ---------------------- 对外接口 ---------------------- #

    def dehaze(self, img_bgr: np.ndarray) -> np.ndarray:
        """
        完整去雾流程，输入输出都是 BGR uint8.
        """
        img = self._to_float(img_bgr)

        dark = self.dark_channel(img)
        A = self.estimate_atmospheric_light(img, dark)
        t = self.estimate_transmission(img, A)
        t_refined = self.refine_transmission(img, t)
        J = self.recover(img, t_refined, A)

        return self._to_uint8(J)

    def __call__(self, img_bgr: np.ndarray) -> np.ndarray:
        return self.dehaze(img_bgr)


# 注意：这个函数就是 framework.py 里通过 get_algorithm_fn 拿到的入口
def dehaze(
    hazy_bgr: np.ndarray,
    patch_size: int = 15,
    omega: float = 0.95,
    t0: float = 0.1,
    use_guided_filter: bool = True,
    guided_radius: int = 40,
    guided_eps: float = 1e-3,
) -> np.ndarray:
    """
    模块级别的 dehaze 接口，供框架调用:

    from .algorithms import dcp as algo
    out = algo.dehaze(hazy_bgr, **params)

    这些参数可以在 config.yaml 里的 algorithms.params 里配置。
    """
    dehazer = DCPDehazer(
        patch_size=patch_size,
        omega=omega,
        t0=t0,
        use_guided_filter=use_guided_filter,
        guided_radius=guided_radius,
        guided_eps=guided_eps,
    )
    return dehazer.dehaze(hazy_bgr)
