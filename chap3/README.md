# GridWorld 动态规划练习：策略评估与策略迭代

本仓库用于实现 Sutton 和 Barto 《Reinforcement Learning: An Introduction》第 3 章“有限马尔可夫决策过程”中的动态规划算法练习。通过本项目深入理解 MDP 中环境建模、Bellman 方程及策略迭代与价值迭代的数值求解方法。

## 项目概述

- **名称**：GridWorld 动态规划：策略评估与策略迭代  
- **环境**：4×4（或 5×5）网格世界  
  - 两个角落为终止状态（terminal），其他格子每步即时奖励 -1  
  - 动作空间为上、下、左、右，环境动力学为确定性转移  

## 项目目标

1. 构建 MDP 环境：状态空间、动作空间、转移概率与即时奖励函数  
2. 实现 **策略评估（Policy Evaluation）**：使用 Bellman 期望方程计算给定策略下的状态价值函数 \(v_\pi\)  
3. 实现 **策略改进（Policy Improvement）**：基于当前价值函数生成新的贪婪策略 \(\pi_{\mathrm{new}}\)  
4. 实现 **策略迭代（Policy Iteration）**：反复执行策略评估和策略改进，直至策略收敛到最优  
5. （可选）实现 **价值迭代（Value Iteration）**：直接应用 Bellman 最优性方程求解最优价值函数，并提取最优策略  

## 算法流程

1. **环境构建**  
    * **States**: 网格中的单元格对应于环境的状态
    * **Actions**: 在每个单元格中，有四种可能的动作：北、南、东和西，这确定地导致 Agent 在网格上各自的方向上移动一个单元格
    * **Rewards**:
      * 特殊状态：
        * 从状态 A 开始，所有4个动作的奖励均为+10，并将Agent带到 A'；
        * 从状态 B 开始，所有动作都产生+5的奖励，并将Agent带到 B'
      * 若执行 action 将导致超出网格边界，则执行后位置不变，但会导致奖励-1；
      * 其他动作的奖励为 0 ；

2. **策略评估（Policy Evaluation）**

   重复应用 Bellman 期望方程：  
   $$
   v(s) \leftarrow \sum_{a}\pi(a\mid s)\,\sum_{s',r}p(s',r\mid s,a)\bigl[r + \gamma\,v(s')\bigr]
   $$
   直到 $\|v_{\mathrm{new}} - v\|_\infty < \epsilon$ 收敛。

3. **策略改进（Policy Improvement）**

   对每个状态 \($s$\)，先计算动作价值：  
   $$
   q(s,a) = \sum_{s',r}p(s',r\mid s,a)\bigl[r + \gamma\,v(s')\bigr]  
   然后更新策略：
   $$

   $$
   \pi_{\mathrm{new}}(s) = \arg\max_{a}\,q(s,a)
   $$

4. **策略迭代（Policy Iteration）**

   按以下步骤循环直至策略不再变化：  
   1. 策略评估  
   2. 策略改进  
   
   最终输出最优策略 $\pi^*$ 与最优价值函数 $v_*$

5. **（可选）价值迭代（Value Iteration）**

   直接应用 Bellman 最优性方程更新：
   $$
   v(s) \leftarrow \max_{a}\sum_{s',r}p(s',r\mid s,a)\bigl[r + \gamma\,v(s')\bigr]
   $$
   迭代至收敛后，再通过
   $$
   \pi^*(s) = \arg\max_{a}\sum_{s',r}p(s',r\mid s,a)\bigl[r + \gamma\,v(s')\bigr]
   $$
   提取最优策略。