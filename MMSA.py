class MMSA(nn.Module):
    def __init__(self, in_size, local_size=5, gamma=2, b=1, local_weight=0.3, num_heads=8):
        super(MMSA, self).__init__()

        # ECA 计算方法
        self.local_size = local_size
        self.gamma = gamma
        self.b = b
        t = int(abs(math.log(in_size, 2) + self.b) / self.gamma)  # eca  gamma=2
        k = t if t % 2 else t + 1
        # 初始化MultiheadAttention
        self.multihead_attn = MultiheadAttention(embed_dim=in_size, num_heads=num_heads)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)
        self.conv_local = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)

        self.local_weight = nn.Parameter(
            torch.Tensor([local_weight])
        )  
        # -----------------------自适应加权融合---------------

        self.local_arv_pool = nn.AdaptiveMaxPool2d(local_size)
        self.global_arv_pool = nn.AdaptiveMaxPool2d(1)

    def forward(self, x):
        local_arv = self.local_arv_pool(x)
        global_arv = self.global_arv_pool(local_arv)

        b, c, m, n = x.shape
        b_local, c_local, m_local, n_local = local_arv.shape
        # 将4D张量转换为3D张量以适应MultiheadAttention
        b, c, h, w = x.size()
        x_reshaped = x.view(b, c, h * w).transpose(1, 2)  # (b, h*w, c)
        # 计算MultiheadAttention
        attn_output, attn_output_weights = self.multihead_attn(query=x_reshaped, key=x_reshaped, value=x_reshaped)
        # 将注意力输出转换回原始的4D形状
        attn_output_reshaped = attn_output.transpose(1, 2).view(b, c, h, w)
        # (b,c,local_size,local_size) -> (b,c,local_size*local_size)-> (b,local_size*local_size,c)-> (b,1,local_size*local_size*c)
        temp_local = local_arv.view(b, c_local, -1).transpose(-1, -2).reshape(b, 1, -1)
        temp_global = global_arv.view(b, c, -1).transpose(-1, -2)

        y_local = self.conv_local(temp_local)
        y_global = self.conv(temp_global)

        # (b,c,local_size,local_size) <- (b,c,local_size*local_size)<-(b,local_size*local_size,c) <- (b,1,local_size*local_size*c)
        y_local_transpose = (
            y_local.reshape(b, self.local_size * self.local_size, c)
            .transpose(-1, -2)
            .view(b, c, self.local_size, self.local_size)
        )
        # y_global_transpose = y_global.view(b, -1).transpose(-1, -2).unsqueeze(-1)
        y_global_transpose = y_global.view(b, -1).unsqueeze(-1).unsqueeze(-1)  # 
        # print(y_global_transpose.size())
        # 反池化
        # print(att_local.size())
        att_local = F.hardsigmoid(y_local_transpose)
        att_global = F.adaptive_avg_pool2d(F.hardsigmoid(y_global_transpose), [self.local_size, self.local_size])
        # print(att_global.size())
        att_all = F.adaptive_avg_pool2d(
            (att_global * (1 - self.local_weight) + (att_local * self.local_weight)), [m, n]
        ) 
        # print(att_all.size())
        x = x * att_all + x
        # 将注意力权重应用于特征图
        x = x + attn_output_reshaped
        return x
