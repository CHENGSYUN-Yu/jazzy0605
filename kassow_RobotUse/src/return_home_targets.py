"""
return_home_targets.py — 回到 Home 位姿兩段目標位姿計算物件（獨立工廠）

職責：
    放置器械並鬆開夾爪後，計算手臂回到 Home 位姿的兩段目標位姿，
    供 TrajectoryPlan 依序執行。

兩段移動：
    1. Lift（Z +200mm）：
       從當前位置原地抬升，避免手臂橫移時碰撞已放置的器械

    2. Home：
       移動到 Home 位姿（448.5, -245.4, 69.7）mm，
       完成後開始下一輪頭部相機偵測

Home 位姿（base frame）：
    x = 448.5 mm
    y = -245.4 mm
    z =   69.7 mm
    yaw = 0.0 deg（預設）

外部使用方式：
    rht = ReturnHomeTargets(lift_z_mm=200.0)
    targets = rht.compute(current_pos, current_rot)

    # targets['lift'] → TrajectoryPlan 第 1 段（Z+200 抬升）
    # targets['home'] → TrajectoryPlan 第 2 段（回到 Home）
"""


# Home 位姿（base frame，mm）
_HOME_X   = 448.5
_HOME_Y   = -245.4
_HOME_Z   =   69.7
_HOME_YAW =    0.0


class ReturnHomeTargets:
    """
    回到 Home 位姿兩段目標位姿計算物件。

    lift_z_mm : 第一段抬升距離（mm），預設 200mm。
    """

    def __init__(self,
                 lift_z_mm: float = 200.0,
                 home_x:    float = _HOME_X,
                 home_y:    float = _HOME_Y,
                 home_z:    float = _HOME_Z,
                 home_yaw:  float = _HOME_YAW):
        self._lift_z   = float(lift_z_mm)
        self._home_x   = float(home_x)
        self._home_y   = float(home_y)
        self._home_z   = float(home_z)
        self._home_yaw = float(home_yaw)

    # ── 計算兩段目標位姿 ──────────────────────────────────────────────────────

    def compute(self, current_pos: list, current_rot: list) -> dict:
        """
        計算回到 Home 的兩段目標位姿。

        參數：
            current_pos : [x, y, z]（mm），當前手臂 TCP 位置
            current_rot : [roll, pitch, yaw]（deg），當前手臂 TCP 姿態

        回傳 dict（各段皆可直接傳入 TrajectoryPlan.plan() 的 target）：
            'lift' : {'x_mm', 'y_mm', 'z_mm', 'yaw_deg'} — Z+200 抬升
            'home' : {'x_mm', 'y_mm', 'z_mm', 'yaw_deg'} — Home 位姿
        """
        cx   = float(current_pos[0])
        cy   = float(current_pos[1])
        cz   = float(current_pos[2])
        cyaw = float(current_rot[2])

        return {
            'lift': {
                'x_mm':    cx,
                'y_mm':    cy,
                'z_mm':    cz + self._lift_z,
                'yaw_deg': cyaw,
            },
            'home': {
                'x_mm':    self._home_x,
                'y_mm':    self._home_y,
                'z_mm':    self._home_z,
                'yaw_deg': self._home_yaw,
            },
        }

    # ── 設定 ──────────────────────────────────────────────────────────────────

    def set_lift_z(self, mm: float):
        self._lift_z = float(mm)

    def set_home(self, x: float, y: float, z: float, yaw: float = 0.0):
        """動態調整 Home 位姿。"""
        self._home_x   = float(x)
        self._home_y   = float(y)
        self._home_z   = float(z)
        self._home_yaw = float(yaw)

    @property
    def lift_z_mm(self) -> float:
        return self._lift_z

    @property
    def home_pose(self) -> dict:
        """取得目前設定的 Home 位姿。"""
        return {
            'x_mm':    self._home_x,
            'y_mm':    self._home_y,
            'z_mm':    self._home_z,
            'yaw_deg': self._home_yaw,
        }
