# src/algorithms/dcp.py
import cv2
import numpy as np


class DCPDehazer:

    def __init__(
        self,
        patch_size: int = 15,
        omega: float = 0.95,
        t0: float = 0.1,
        use_guided_filter: bool = True,
        guided_radius: int = 40,
        guided_eps: float = 1e-3,
    ):

        if patch_size % 2 == 0:
            raise ValueError("patch_size must be odd.")
        self.patch_size = patch_size
        self.omega = omega
        self.t0 = t0
        self.use_guided_filter = use_guided_filter
        self.guided_radius = guided_radius
        self.guided_eps = guided_eps



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



    def dark_channel(self, img: np.ndarray) -> np.ndarray:

        min_per_pixel = np.min(img, axis=2)

        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (self.patch_size, self.patch_size)
        )
        dark = cv2.erode(min_per_pixel, kernel)
        return dark

    def estimate_atmospheric_light(
        self, img: np.ndarray, dark: np.ndarray
    ) -> np.ndarray:

        h, w = dark.shape
        n_pixels = h * w
        n_search = max(int(n_pixels * 0.001), 1)  

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

        normed = img / A.reshape(1, 1, 3)
        normed_dark = self.dark_channel(normed)
        t = 1.0 - self.omega * normed_dark
        return t

    def refine_transmission(self, img: np.ndarray, t: np.ndarray) -> np.ndarray:

        if not self.use_guided_filter:
            return t

        # 用灰度图作为引导图
        gray = cv2.cvtColor(self._to_uint8(img), cv2.COLOR_BGR2GRAY).astype(
            np.float32
        ) / 255.0

        try:

            guided = cv2.ximgproc.guidedFilter(
                guide=gray,
                src=t.astype(np.float32),
                radius=self.guided_radius,
                eps=self.guided_eps,
            )
            return guided
        except Exception:
            ksize = max(3, self.guided_radius // 2 * 2 + 1)  # 确保为奇数
            return cv2.blur(t.astype(np.float32), (ksize, ksize))

    def recover(self, img: np.ndarray, t: np.ndarray, A: np.ndarray) -> np.ndarray:
        t = np.clip(t, self.t0, 1.0)
        t = t[:, :, np.newaxis]  # (H,W,1)

        J = (img - A.reshape(1, 1, 3)) / t + A.reshape(1, 1, 3)
        J = np.clip(J, 0.0, 1.0)
        return J


    def dehaze(self, img_bgr: np.ndarray) -> np.ndarray:

        img = self._to_float(img_bgr)

        dark = self.dark_channel(img)
        A = self.estimate_atmospheric_light(img, dark)
        t = self.estimate_transmission(img, A)
        t_refined = self.refine_transmission(img, t)
        J = self.recover(img, t_refined, A)

        return self._to_uint8(J)

    def __call__(self, img_bgr: np.ndarray) -> np.ndarray:
        return self.dehaze(img_bgr)


def dehaze(
    hazy_bgr: np.ndarray,
    patch_size: int = 15,
    omega: float = 0.95,
    t0: float = 0.1,
    use_guided_filter: bool = True,
    guided_radius: int = 40,
    guided_eps: float = 1e-3,
) -> np.ndarray:

    dehazer = DCPDehazer(
        patch_size=patch_size,
        omega=omega,
        t0=t0,
        use_guided_filter=use_guided_filter,
        guided_radius=guided_radius,
        guided_eps=guided_eps,
    )
    return dehazer.dehaze(hazy_bgr)
