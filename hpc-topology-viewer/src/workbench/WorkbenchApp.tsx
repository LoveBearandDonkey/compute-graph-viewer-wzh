import { ClusterView } from '../view/ClusterView';
import { SceneVisualProfileContext } from '../scene/visual-profile';

// 统一驾驶舱直接建立在原工作台之上：workbench.html 仍渲染 ClusterView（chrome=workbench），
// 保留其全部功能/布局/风格/L0-L7 层级与浅色主题。新增能力（时空折叠魔方 + 动态监控双模式 +
// 集合通信深潜）作为增量融进工作台的「联动控制台」（ConsoleView，仅在 workbench profile 下启用，
// index.html 主应用不受影响）。
export function WorkbenchApp() {
  return (
    <SceneVisualProfileContext.Provider value="opRankTime">
      <ClusterView chrome="workbench" />
    </SceneVisualProfileContext.Provider>
  );
}
