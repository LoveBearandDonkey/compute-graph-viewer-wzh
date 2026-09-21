(function defineDistributedCommunicationMock(global) {
  'use strict';

  const rankIds = [40, 41, 42, 43, 44, 45, 46, 47];
  const rankSlots = new Map(rankIds.map((rank, slot) => [rank, slot]));
  const ranks = rankIds.map((rank, slot) => ({
    rank,
    slot,
    node: 'server-05',
  }));

  const routes = rankIds.flatMap((sourceRank) => [0, 1, 3].map((offset, routeIndex) => {
    const sourceSlot = rankSlots.get(sourceRank);
    const targetSlot = (sourceSlot + offset) % rankIds.length;
    const targetRank = rankIds[targetSlot];
    const hotspot = targetRank === 42 ? 96 : 0;
    const value = 86 + ((sourceRank * 31 + routeIndex * 47) % 92) + hotspot;
    return [sourceRank, targetRank, `e${targetSlot * 32 + routeIndex}`, value];
  }));

  const chunkOffsets = [0, 1.8, 3.7, 5.3];
  function engineFor(code, targetSide = false) {
    const normalized = code % (targetSide ? 12 : 10);
    if (normalized < (targetSide ? 7 : 6)) return 'aic';
    if (normalized < (targetSide ? 10 : 9)) return 'aiv';
    return null;
  }

  const events = routes.flatMap(([sourceRank, targetRank, targetChild, total], routeIndex) => {
    const base = Math.floor(total / chunkOffsets.length);
    return chunkOffsets.map((offset, chunkIndex) => {
      const value = chunkIndex === chunkOffsets.length - 1
        ? total - base * (chunkOffsets.length - 1)
        : base;
      const sameNode = true;
      return {
        id: `dispatch-${routeIndex}-${chunkIndex}`,
        timestamp: 116 + routeIndex % 5 * 6.2 + offset,
        duration: 0.7 + (routeIndex + chunkIndex) % 5 * 0.31,
        value,
        sourceRank,
        targetRank,
        sourceEngine: engineFor(routeIndex * 3 + chunkIndex + targetRank),
        targetEngine: engineFor(routeIndex * 2 + chunkIndex * 3 + sourceRank, true),
        targetExpert: targetChild,
        streamId: `comm-${sourceRank}-${routeIndex % 3}`,
        taskId: `a2av-${sourceRank}-${targetRank}-${chunkIndex}`,
        transport: sourceRank === targetRank ? 'local' : sameNode ? 'hccs' : 'roce',
        source: 'mock',
      };
    });
  });

  function bounds(sourceEvents, fallbackStart, fallbackEnd) {
    if (!sourceEvents.length) return { start: fallbackStart, end: fallbackEnd };
    return {
      start: Math.min(...sourceEvents.map((event) => event.timestamp)),
      end: Math.max(...sourceEvents.map((event) => event.timestamp + event.duration)),
    };
  }

  const dispatchRankRows = ranks.flatMap(({ rank, slot }) => {
    const rankEvents = events.filter((event) => event.targetRank === rank);
    const range = bounds(rankEvents, 116 + rank, 132 + rank);
    return [
      {
        id: `dispatch-rank-${rank}`,
        parentId: 'dispatch-collective',
        depth: 3,
        label: `Rank ${rank}`,
        labelZh: `Rank ${rank}`,
        events: [{
          id: `dispatch-rank-${rank}-aggregate`,
          start: range.start,
          end: range.end,
          label: `R${rank} receive`,
          labelZh: `R${rank} 接收`,
          displayName: `Rank ${rank} Dispatch Receive`,
          displayNameZh: `Rank ${rank} 分发接收`,
          kind: 'rank-transfer',
          status: rank === 42 ? 'hot' : 'mock',
          colorSlot: slot,
          transactionId: `dispatch-rank-${rank}`,
          sourceEventIds: rankEvents.map((event) => event.id),
        }],
      },
      {
        id: `dispatch-rank-${rank}-tasks`,
        parentId: `dispatch-rank-${rank}`,
        depth: 4,
        label: 'Transfer tasks',
        labelZh: '传输任务',
        events: rankEvents.map((event) => ({
          id: `timeline-${event.id}`,
          start: event.timestamp,
          end: event.timestamp + event.duration,
          label: `R${event.sourceRank} → R${event.targetRank}`,
          displayName: `Dispatch task ${event.taskId}`,
          displayNameZh: `分发任务 ${event.taskId}`,
          kind: 'task',
          status: event.source,
          colorSlot: rankSlots.get(event.sourceRank),
          transactionId: event.taskId,
          sourceEventIds: [event.id],
          objectRefs: {
            sourceRank: event.sourceRank,
            targetRank: event.targetRank,
            sourceEngine: event.sourceEngine,
            targetEngine: event.targetEngine,
            targetExpert: event.targetExpert,
          },
        })),
      },
    ];
  });

  const combineRankRows = ranks.flatMap(({ rank, slot }) => {
    const rankEvents = events.filter((event) => event.targetRank === rank);
    const sourceEventIds = rankEvents.flatMap((event) => [event.id, `combine-${event.id}`]);
    const start = 151 + slot * 0.55;
    const end = rank === 42 ? 169 : 165.5 + slot * 0.45;
    return [
      {
        id: `combine-rank-${rank}`,
        parentId: 'combine-collective',
        depth: 3,
        label: `Rank ${rank}`,
        labelZh: `Rank ${rank}`,
        events: [{
          id: `combine-rank-${rank}-aggregate`,
          start,
          end,
          label: `R${rank} return`,
          labelZh: `R${rank} 回传`,
          displayName: `Rank ${rank} Combine Return`,
          displayNameZh: `Rank ${rank} 回传`,
          kind: 'rank-transfer',
          status: rank === 42 ? 'tail' : 'mock',
          colorSlot: slot,
          transactionId: `combine-rank-${rank}`,
          sourceEventIds,
        }],
      },
      {
        id: `combine-rank-${rank}-tasks`,
        parentId: `combine-rank-${rank}`,
        depth: 4,
        label: 'Return tasks',
        labelZh: '回传任务',
        events: rankEvents.slice(0, 8).map((event, index) => ({
          id: `timeline-combine-${event.id}`,
          start: start + index * 0.72,
          end: Math.min(end, start + index * 0.72 + event.duration),
          label: `R${rank} → R${event.sourceRank}`,
          displayName: `Combine task ${event.taskId}`,
          displayNameZh: `回传任务 ${event.taskId}`,
          kind: 'task',
          status: 'mock',
          colorSlot: rankSlots.get(event.sourceRank),
          transactionId: `combine-${event.taskId}`,
          sourceEventIds: [event.id, `combine-${event.id}`],
        })),
      },
    ];
  });

  const timelineHierarchy = {
    id: 'layer-27-communication-lifecycle',
    ariaLabel: 'Layer 27 hierarchical communication timeline',
    ariaLabelZh: 'Layer 27 分层通信时间线',
    unit: 'ms',
    defaultExpandedIds: ['layer-27', 'dispatch', 'dispatch-collective', 'dispatch-rank-2'],
    rows: [
      {
        id: 'layer-27', label: 'Layer 27 lifecycle', labelZh: 'Layer 27 生命周期', depth: 0,
        events: [{ id: 'layer-27-window', start: 108, end: 169, label: 'Layer 27', displayName: 'Layer 27 Communication Lifecycle', displayNameZh: 'Layer 27 通信生命周期', kind: 'summary', status: 'derived', colorSlot: 'summary', sourceEventIds: events.map((event) => event.id) }],
      },
      {
        id: 'router', parentId: 'layer-27', label: 'Router', labelZh: '路由', depth: 1,
        events: [{ id: 'router-27', start: 108, end: 117, label: 'top-k routing', labelZh: 'top-k 路由', displayName: 'Layer 27 Router', displayNameZh: 'Layer 27 路由', kind: 'phase', status: 'derived', colorSlot: 0 }],
      },
      {
        id: 'dispatch', parentId: 'layer-27', label: 'Dispatch', labelZh: '分发', depth: 1,
        events: [{ id: 'dispatch-phase', start: 115, end: 137, label: 'Dispatch', labelZh: '分发', displayName: 'EP8 Dispatch Phase', displayNameZh: 'EP8 分发阶段', kind: 'phase', status: 'mock', colorSlot: 1, sourceEventIds: events.map((event) => event.id) }],
      },
      {
        id: 'dispatch-collective', parentId: 'dispatch', label: 'AllToAllv collective', labelZh: 'AllToAllv 集合通信', depth: 2,
        events: [{ id: 'dispatch-a2av', start: 115, end: 137, label: 'AllToAllv', displayName: 'EP8 Dispatch AllToAllv', displayNameZh: 'EP8 分发 AllToAllv', kind: 'collective', status: 'mock', colorSlot: 1, transactionId: 'dispatch-a2av-layer-27', sourceEventIds: events.map((event) => event.id) }],
      },
      ...dispatchRankRows,
      {
        id: 'compute', parentId: 'layer-27', label: 'Expert compute', labelZh: '专家计算', depth: 1,
        events: [{ id: 'expert-compute', start: 132, end: 153, label: 'Expert compute', labelZh: '专家计算', displayName: 'Hot Expert Compute', displayNameZh: '热点专家计算', kind: 'phase', status: 'mock', colorSlot: 2 }],
      },
      {
        id: 'compute-e64', parentId: 'compute', label: 'E64', depth: 2,
        events: [{ id: 'expert-e64', start: 132, end: 153, label: 'E64 compute', labelZh: 'E64 计算', displayName: 'E64 Expert Compute', displayNameZh: 'E64 专家计算', kind: 'compute', status: 'hot', colorSlot: 2, transactionId: 'expert-e64' }],
      },
      {
        id: 'compute-e65', parentId: 'compute', label: 'E65', depth: 2,
        events: [{ id: 'expert-e65', start: 133.4, end: 151.8, label: 'E65 compute', labelZh: 'E65 计算', displayName: 'E65 Expert Compute', displayNameZh: 'E65 专家计算', kind: 'compute', status: 'hot', colorSlot: 3, transactionId: 'expert-e65' }],
      },
      {
        id: 'wait', parentId: 'layer-27', label: 'Exposed wait', labelZh: '暴露等待', depth: 1,
        events: [{ id: 'wait-r42', start: 137, end: 141.8, label: 'Wait for R42', labelZh: '等待 R42', displayName: 'Exposed Rank Wait', displayNameZh: 'Rank 暴露等待', kind: 'wait', status: 'derived', colorSlot: 'wait', dominantCounter: '4.8 ms exposed', transactionId: 'wait-chain-r42', sourceEventIds: events.filter((event) => event.targetRank === 42).map((event) => event.id) }],
      },
      {
        id: 'combine', parentId: 'layer-27', label: 'Combine', labelZh: '回传', depth: 1,
        events: [{ id: 'combine-phase', start: 151, end: 169, label: 'Combine', labelZh: '回传', displayName: 'EP8 Combine Phase', displayNameZh: 'EP8 回传阶段', kind: 'phase', status: 'mock', colorSlot: 3, sourceEventIds: events.flatMap((event) => [event.id, `combine-${event.id}`]) }],
      },
      {
        id: 'combine-collective', parentId: 'combine', label: 'AllToAllv collective', labelZh: 'AllToAllv 集合通信', depth: 2,
        events: [{ id: 'combine-a2av', start: 151, end: 169, label: 'AllToAllv', displayName: 'EP8 Combine AllToAllv', displayNameZh: 'EP8 回传 AllToAllv', kind: 'collective', status: 'mock', colorSlot: 3, transactionId: 'combine-a2av-layer-27', sourceEventIds: events.flatMap((event) => [event.id, `combine-${event.id}`]) }],
      },
      ...combineRankRows,
    ],
  };

  const phases = {
    router: {
      label: 'Router',
      severity: 'MEDIUM',
      title: 'Layer 27 Router sends 2.1× mean tokens to E64/E65',
      impact: 'The skew consumes most of R42 receive capacity before Dispatch begins.',
      next: 'Inspect the token sources contributing to E64/E65.',
    },
    dispatch: {
      label: 'Dispatch',
      severity: 'HIGH',
      title: 'MoE Layer 27 Dispatch is on the critical path',
      impact: 'E64/E65 on R42 receive 2.1× mean load, exposing 4.8 ms wait with 6% buffer headroom.',
      next: 'Trace E64/E65 token sources and compare HCCS peer traffic inside Server 05.',
    },
    compute: {
      label: 'Expert compute',
      severity: 'HIGH',
      title: 'Hot experts extend the tail of Layer 27 compute',
      impact: 'The other seven cards complete earlier and wait for E64/E65 on R42 before Combine can start.',
      next: 'Separate router skew from expert placement and kernel variance.',
    },
    combine: {
      label: 'Combine',
      severity: 'MEDIUM',
      title: 'Combine inherits the Dispatch imbalance',
      impact: 'The return path starts late because R42 experts finish last; payload size alone is not the root cause.',
      next: 'Verify whether reducing Dispatch skew removes the Combine delay.',
    },
  };

  global.DistributedCommunicationMock = {
    context: {
      run: 'pangu-flash-train-0421',
      baseline: 'pangu-flash-train-0418',
      step: 'Step 18420',
      stage: 'Forward · Layer 27',
      group: '128 ranks · 16 servers × 8 cards · EP8',
      evidence: '3/3 complete',
      source: 'MOCK SCENARIO',
    },
    timeRange: { min: 96, max: 176, step: 1 },
    defaultWindow: { start: 115, end: 152 },
    ranks,
    events,
    timelineHierarchy,
    phases,
  };
})(window);
