#pragma once
// 日志开关
namespace training::dbg
{
    extern bool io;   // [IO] yaw/pitch/bullet、[read] nullopt、[Write]
    extern bool det;  // [det] 每帧检测耗时、模型/模板加载
    extern bool rune; // [rune] 圆心 r/w、模板分数
    extern bool gui;  // [gui] 按键切换提示

    extern bool cv_filter;  // 演示：同时运行匀速卡尔曼（CV），供与 EKF 对比；关闭时零开销
    extern bool demo_write; // 演示：无视算法输出，向云台发出按正弦函数运动的指令

    extern bool truth;    // 返回仿真器数据，严禁在实车上开启
    extern bool pred_log; // 输出预测值 & 真实值日志（真实值依赖 truth）

    // [aim] 第五讲：云台角、目标偏差、弹道补偿、最终指令（每 15 帧一行）
    extern bool aim; // 输出预测值 & 真实值日志（真实值依赖 truth）
} // namespace training::dbg