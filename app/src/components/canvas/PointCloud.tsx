import React, { ReactNode, useCallback, useMemo, useState } from "react";
import { css } from "@emotion/react";

import { theme } from "@arizeai/components";
import {
  Axes,
  getThreeDimensionalBounds,
  LassoSelect,
  PointBaseProps,
  ThreeDimensionalBounds,
  ThreeDimensionalCanvas,
  ThreeDimensionalControls,
} from "@arizeai/point-cloud";

import { usePointCloudStore } from "@phoenix/store";

import { CanvasMode, CanvasModeRadioGroup } from "./CanvasModeRadioGroup";
import { getPointColorGroup } from "./colorGroups";
import { ControlPanel } from "./ControlPanel";
import { PointCloudClusters } from "./PointCloudClusters";
import { PointCloudPoints } from "./PointCloudPoints";
import { ThreeDimensionalPointItem } from "./types";
import { ClusterItem } from "./types";

const RADIUS_BOUNDS_3D_DIVISOR = 400;
const CLUSTER_POINT_RADIUS_MULTIPLIER = 6;
const BOUNDS_3D_ZOOM_PADDING_FACTOR = 0.5;

export interface PointCloudProps {
  primaryData: ThreeDimensionalPointItem[];
  referenceData: ThreeDimensionalPointItem[] | null;
  clusters: readonly ClusterItem[];
}

interface ProjectionProps extends PointCloudProps {
  canvasMode: CanvasMode;
}

const CONTROL_PANEL_WIDTH = 300;
/**
 * Displays the tools available on the point cloud
 * E.g. move vs select
 */
function CanvasTools(props: {
  canvasMode: CanvasMode;
  onCanvasModeChange: (mode: CanvasMode) => void;
}) {
  const { canvasMode, onCanvasModeChange } = props;
  return (
    <div
      css={css`
        position: absolute;
        /* left: ${CONTROL_PANEL_WIDTH + 2 * theme.spacing.margin8}px; */
        left: ${theme.spacing.margin8}px;
        top: ${theme.spacing.margin8}px;
        z-index: 1;
        display: flex;
        flex-direction: row;
        gap: ${theme.spacing.margin8}px;
      `}
    >
      <CanvasModeRadioGroup mode={canvasMode} onChange={onCanvasModeChange} />
    </div>
  );
}

// eslint-disable-next-line @typescript-eslint/no-unused-vars
function SelectionControlPanel({ selectedIds }: { selectedIds: Set<string> }) {
  return (
    <ControlPanel position="top-right" width={CONTROL_PANEL_WIDTH}>
      <ul>
        {[...selectedIds].map((id) => (
          <li key={id}>{id}</li>
        ))}
      </ul>
    </ControlPanel>
  );
}

function CanvasWrap({ children }: { children: ReactNode }) {
  return (
    <div
      css={css`
        flex: 1 1 auto;
        height: 100%;
        position: relative;
        background-color: black;
      `}
    >
      {children}
    </div>
  );
}

export function PointCloud(props: PointCloudProps) {
  const [canvasMode, setCanvasMode] = useState<CanvasMode>(CanvasMode.move);

  return (
    <CanvasWrap>
      <CanvasTools canvasMode={canvasMode} onCanvasModeChange={setCanvasMode} />
      <Projection canvasMode={canvasMode} {...props} />
    </CanvasWrap>
  );
}

function Projection(props: ProjectionProps) {
  const { primaryData, referenceData, clusters, canvasMode } = props;

  const coloringStrategy = usePointCloudStore(
    (state) => state.coloringStrategy
  );
  const selectedPointIds = usePointCloudStore(
    (state) => state.selectedPointIds
  );
  const setSelectedPointIds = usePointCloudStore(
    (state) => state.setSelectedPointIds
  );
  const selectedClusterId = usePointCloudStore(
    (state) => state.selectedClusterId
  );
  const setSelectedClusterId = usePointCloudStore(
    (state) => state.setSelectedClusterId
  );
  const pointGroupColors = usePointCloudStore(
    (state) => state.pointGroupColors
  );
  const pointGroupVisibility = usePointCloudStore(
    (state) => state.pointGroupVisibility
  );

  // AutoRotate the canvas on initial load
  const [autoRotate, setAutoRotate] = useState<boolean>(true);

  const allPoints = useMemo(() => {
    return [...primaryData, ...(referenceData || [])];
  }, [primaryData, referenceData]);

  const bounds = useMemo(() => {
    return getThreeDimensionalBounds(allPoints.map((p) => p.position));
  }, [allPoints]);

  const radius =
    (bounds.maxX - bounds.minX + (bounds.maxY - bounds.minY)) /
    2 /
    RADIUS_BOUNDS_3D_DIVISOR;

  const clusterPointRadius = radius * CLUSTER_POINT_RADIUS_MULTIPLIER;

  const isMoveMode = canvasMode === CanvasMode.move;

  const pointToColorGroup = useMemo(() => {
    const pointToGroup = getPointColorGroup(coloringStrategy);
    return allPoints.reduce((acc, point) => {
      acc[point.metaData.id] = pointToGroup(point);
      return acc;
    }, {} as Record<string, string>);
  }, [coloringStrategy, allPoints]);

  // Color the points by their corresponding group
  const colorFn = useCallback(
    (point: PointBaseProps) => {
      const group = pointToColorGroup[point.metaData.id];
      return pointGroupColors[group];
    },
    [pointGroupColors, pointToColorGroup]
  );

  // Filter the points by the group visibility
  const filteredPrimaryData = useMemo(() => {
    return primaryData.filter((point) => {
      const group = pointToColorGroup[point.metaData.id];
      return pointGroupVisibility[group];
    });
  }, [primaryData, pointToColorGroup, pointGroupVisibility]);

  const filteredReferenceData = useMemo(() => {
    if (!referenceData) {
      return null;
    }
    return referenceData.filter((point) => {
      const group = pointToColorGroup[point.metaData.id];
      return pointGroupVisibility[group];
    });
  }, [referenceData, pointToColorGroup, pointGroupVisibility]);
  return (
    <ThreeDimensionalCanvas camera={{ position: [0, 0, 10] }}>
      <ThreeDimensionalControls
        autoRotate={autoRotate}
        autoRotateSpeed={2}
        enableRotate={isMoveMode}
        enablePan={isMoveMode}
        onEnd={() => {
          // Turn off auto rotate when the user interacts with the canvas
          setAutoRotate(false);
        }}
      />
      <ThreeDimensionalBounds
        bounds={bounds}
        boundsZoomPaddingFactor={BOUNDS_3D_ZOOM_PADDING_FACTOR}
      >
        <LassoSelect
          points={allPoints}
          onChange={(selection) => {
            setSelectedPointIds(new Set(selection.map((s) => s.metaData.id)));
            setSelectedClusterId(null);
          }}
          enabled={canvasMode === CanvasMode.select}
        />
        <Axes size={(bounds.maxX - bounds.minX) / 4} />
        <PointCloudPoints
          primaryData={filteredPrimaryData}
          referenceData={filteredReferenceData}
          selectedIds={selectedPointIds}
          color={colorFn}
          radius={radius}
        />
        <PointCloudClusters
          clusters={clusters}
          points={allPoints}
          selectedClusterId={selectedClusterId}
          radius={clusterPointRadius}
        />
      </ThreeDimensionalBounds>
    </ThreeDimensionalCanvas>
  );
}