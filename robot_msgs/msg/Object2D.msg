# 2D 物体检测结果
std_msgs/Header header     # 时间戳
string label               # 物体类别
float32 confidence         # 置信度 [0-1]
float32 center_x           # 中心点 x 坐标 [pixels]
float32 center_y           # 中心点 y 坐标 [pixels]
float32 width              # 宽度 [pixels]
float32 height             # 高度 [pixels]