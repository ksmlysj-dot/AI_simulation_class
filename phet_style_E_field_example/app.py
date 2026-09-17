from __future__ import annotations

import re

import numpy as np
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, ctx, dcc, html

from physics import field_and_potential, probe_measurement


# ------------------------------------------------------------
# Classroom defaults
# ------------------------------------------------------------
DEFAULT_POSITIONS = {
    "x1": -1.5,
    "y1": 1.0,
    "x2": 1.5,
    "y2": 0.0,
}
CHARGE_RADIUS = 0.50
AXIS_LIMIT = 5.0
POSITION_LIMIT = AXIS_LIMIT - CHARGE_RADIUS - 0.05


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, float(value)))


def charge_style(q: float) -> tuple[str, str]:
    """Return fill and border colors according to charge sign."""
    if q > 0:
        return "rgba(220, 60, 60, 0.82)", "rgba(145, 25, 25, 1)"
    if q < 0:
        return "rgba(55, 105, 210, 0.82)", "rgba(30, 65, 145, 1)"
    return "rgba(130, 130, 130, 0.75)", "rgba(80, 80, 80, 1)"


def _box_from_center(x: float, y: float) -> dict[str, float]:
    return {
        "x0": x - CHARGE_RADIUS,
        "x1": x + CHARGE_RADIUS,
        "y0": y - CHARGE_RADIUS,
        "y1": y + CHARGE_RADIUS,
    }


def _center_from_box(box: dict[str, float]) -> tuple[float, float]:
    x = 0.5 * (box["x0"] + box["x1"])
    y = 0.5 * (box["y0"] + box["y1"])
    return (
        clamp(x, -POSITION_LIMIT, POSITION_LIMIT),
        clamp(y, -POSITION_LIMIT, POSITION_LIMIT),
    )


def update_positions_from_relayout(
    relayout_data: dict | None,
    positions: dict[str, float],
) -> dict[str, float]:
    """Read dragged Plotly circle positions from relayoutData.

    Plotly can also resize an editable shape.  For this classroom app we only
    keep its new CENTER and redraw it at the original radius, so the charge
    always remains a fixed-size circle.
    """
    if not relayout_data:
        return dict(positions)

    boxes = [
        _box_from_center(positions["x1"], positions["y1"]),
        _box_from_center(positions["x2"], positions["y2"]),
    ]

    # Plotly sometimes sends the complete shape list.
    if "shapes" in relayout_data:
        shapes = relayout_data["shapes"]
        for i in range(min(2, len(shapes))):
            for key in ("x0", "x1", "y0", "y1"):
                if key in shapes[i]:
                    boxes[i][key] = float(shapes[i][key])

    # When an existing shape is edited, Plotly often sends only changed keys:
    # "shapes[0].x0", "shapes[0].x1", ...
    pattern = re.compile(r"shapes\[(\d+)\]\.(x0|x1|y0|y1)")
    for key, value in relayout_data.items():
        match = pattern.fullmatch(key)
        if match:
            shape_index = int(match.group(1))
            coordinate = match.group(2)
            if shape_index < 2:
                boxes[shape_index][coordinate] = float(value)

    x1, y1 = _center_from_box(boxes[0])
    x2, y2 = _center_from_box(boxes[1])

    return {"x1": x1, "y1": y1, "x2": x2, "y2": y2}


def build_figure(
    q1: float,
    q2: float,
    positions: dict[str, float],
    probe_x: float,
    probe_y: float,
    show_vectors: bool,
    show_potential: bool,
) -> go.Figure:
    charges = [
        (q1, positions["x1"], positions["y1"]),
        (q2, positions["x2"], positions["y2"]),
    ]

    axis = np.linspace(-AXIS_LIMIT, AXIS_LIMIT, 121)
    X, Y = np.meshgrid(axis, axis)
    _, _, V = field_and_potential(X, Y, charges)

    fig = go.Figure()

    # Equipotential contours
    if show_potential:
        clip = max(1.0, float(np.percentile(np.abs(V), 92)))
        fig.add_trace(
            go.Contour(
                x=axis,
                y=axis,
                z=np.clip(V, -clip, clip),
                contours=dict(coloring="lines", showlabels=False),
                line=dict(width=1),
                colorbar=dict(title="V"),
                hovertemplate="x=%{x:.2f}<br>y=%{y:.2f}<br>V=%{z:.3f}<extra></extra>",
                name="Equipotential",
            )
        )

    # Electric-field vectors
    if show_vectors:
        coarse = np.linspace(-4.6, 4.6, 19)
        XV, YV = np.meshgrid(coarse, coarse)
        EVx, EVy, _ = field_and_potential(XV, YV, charges)

        mag = np.hypot(EVx, EVy)
        ux = EVx / (mag + 1e-12)
        uy = EVy / (mag + 1e-12)

        arrow_length = 0.32
        xs: list[float | None] = []
        ys: list[float | None] = []
        hx: list[float] = []
        hy: list[float] = []
        angles: list[float] = []

        for x0, y0, dx, dy in zip(
            XV.ravel(), YV.ravel(), ux.ravel(), uy.ravel()
        ):
            near_charge = any(
                (x0 - cx) ** 2 + (y0 - cy) ** 2 < 0.42**2
                for _, cx, cy in charges
            )
            if near_charge:
                continue

            x_end = x0 + arrow_length * dx
            y_end = y0 + arrow_length * dy
            xs.extend([x0, x_end, None])
            ys.extend([y0, y_end, None])
            hx.append(x_end)
            hy.append(y_end)
            angles.append(float(np.degrees(np.arctan2(dy, dx)) - 90.0))

        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="lines",
                line=dict(width=1),
                hoverinfo="skip",
                name="E field",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=hx,
                y=hy,
                mode="markers",
                marker=dict(symbol="triangle-up", size=5, angle=angles),
                hoverinfo="skip",
                showlegend=False,
            )
        )

    # Probe
    fig.add_trace(
        go.Scatter(
            x=[probe_x],
            y=[probe_y],
            mode="markers",
            marker=dict(symbol="x", size=14, line=dict(width=2)),
            name="Probe",
            hovertemplate="Probe<br>x=%{x:.2f}<br>y=%{y:.2f}<extra></extra>",
        )
    )

    # Draggable charge circles are Plotly layout shapes.
    for index, (q, x, y) in enumerate(charges, start=1):
        fill, border = charge_style(q)
        sign = "+" if q > 0 else "−" if q < 0 else "0"

        fig.add_shape(
            type="circle",
            xref="x",
            yref="y",
            x0=x - CHARGE_RADIUS,
            x1=x + CHARGE_RADIUS,
            y0=y - CHARGE_RADIUS,
            y1=y + CHARGE_RADIUS,
            fillcolor=fill,
            line=dict(color=border, width=2),
            editable=True,
            name=f"q{index}",
        )
        fig.add_annotation(
            x=x,
            y=y,
            text=f"<b>{sign}</b>",
            showarrow=False,
            font=dict(size=20),
        )
        fig.add_annotation(
            x=x,
            y=y + 0.55,
            text=f"q{index}={q:+.1f}",
            showarrow=False,
            font=dict(size=12),
        )

    fig.update_layout(
        height=690,
        margin=dict(l=25, r=25, t=20, b=20),
        xaxis=dict(
            title="x",
            range=[-AXIS_LIMIT, AXIS_LIMIT],
            scaleanchor="y",
            scaleratio=1,
            fixedrange=True,
            zeroline=True,
        ),
        yaxis=dict(
            title="y",
            range=[-AXIS_LIMIT, AXIS_LIMIT],
            fixedrange=True,
            zeroline=True,
        ),
        legend=dict(orientation="h", y=1.03),
        hovermode="closest",
        uirevision="electric-field-classroom",
    )
    return fig


# ------------------------------------------------------------
# Dash UI
# ------------------------------------------------------------
app = Dash(__name__)
app.title = "Electric Field Playground"

CARD_STYLE = {
    "border": "1px solid #d8d8d8",
    "borderRadius": "10px",
    "padding": "14px",
    "marginBottom": "12px",
}

app.layout = html.Div(
    [
        dcc.Store(id="positions", data=DEFAULT_POSITIONS),

        html.H1("⚡ Electric Field Playground", style={"marginBottom": "4px"}),
        html.P(
            "전하 원을 클릭한 뒤 드래그해서 위치를 바꿔 보세요. "
            "전기장과 등전위선은 Python에서 다시 계산됩니다.",
            style={"marginTop": "0"},
        ),

        html.Div(
            [
                # Left controls
                html.Div(
                    [
                        html.Div(
                            [
                                html.H3("전하량"),
                                html.Label("q₁"),
                                dcc.Slider(
                                    id="q1",
                                    min=-5,
                                    max=5,
                                    step=0.5,
                                    value=2.0,
                                    marks={-5: "-5", 0: "0", 5: "+5"},
                                ),
                                html.Br(),
                                html.Label("q₂"),
                                dcc.Slider(
                                    id="q2",
                                    min=-5,
                                    max=5,
                                    step=0.5,
                                    value=-2.0,
                                    marks={-5: "-5", 0: "0", 5: "+5"},
                                ),
                            ],
                            style=CARD_STYLE,
                        ),

                        html.Div(
                            [
                                html.H3("표시"),
                                dcc.Checklist(
                                    id="display-options",
                                    options=[
                                        {"label": " 전기장 벡터", "value": "vectors"},
                                        {"label": " 등전위선", "value": "potential"},
                                    ],
                                    value=["vectors", "potential"],
                                    labelStyle={"display": "block", "marginBottom": "8px"},
                                ),
                            ],
                            style=CARD_STYLE,
                        ),

                        html.Div(
                            [
                                html.H3("측정 Probe"),
                                html.Label("Probe x"),
                                dcc.Slider(
                                    id="probe-x",
                                    min=-4.8,
                                    max=4.8,
                                    step=0.1,
                                    value=0.0,
                                    marks={-4: "-4", 0: "0", 4: "4"},
                                ),
                                html.Br(),
                                html.Label("Probe y"),
                                dcc.Slider(
                                    id="probe-y",
                                    min=-4.8,
                                    max=4.8,
                                    step=0.1,
                                    value=2.0,
                                    marks={-4: "-4", 0: "0", 4: "4"},
                                ),
                                html.Div(id="probe-readout", style={"marginTop": "14px"}),
                            ],
                            style=CARD_STYLE,
                        ),

                        html.Button(
                            "위치 초기화",
                            id="reset",
                            n_clicks=0,
                            style={"padding": "9px 14px", "cursor": "pointer"},
                        ),
                    ],
                    style={"minWidth": "260px", "flex": "0 0 300px"},
                ),

                # Main graph
                html.Div(
                    [
                        dcc.Graph(
                            id="field-graph",
                            config={
                                "displaylogo": False,
                                "editable": False,
                                "edits": {"shapePosition": True},
                                "modeBarButtonsToRemove": [
                                    "zoom2d",
                                    "pan2d",
                                    "select2d",
                                    "lasso2d",
                                    "autoScale2d",
                                ],
                            },
                            style={"width": "100%"},
                        ),
                        html.Div(
                            id="position-readout",
                            style={
                                "fontFamily": "monospace",
                                "fontSize": "14px",
                                "margin": "4px 0 12px 8px",
                            },
                        ),
                    ],
                    style={"flex": "1 1 700px", "minWidth": "0"},
                ),
            ],
            style={
                "display": "flex",
                "gap": "18px",
                "alignItems": "flex-start",
                "flexWrap": "wrap",
            },
        ),

        html.H3("생각해 보기"),
        html.Ol(
            [
                html.Li("두 전하를 가까이/멀리 드래그하면 중앙의 |E|는 어떻게 변할까?"),
                html.Li("q₁과 q₂의 부호를 같게 만들면 등전위선은 어떻게 달라질까?"),
                html.Li("Probe를 전기장이 0에 가까운 위치로 옮겨 보자."),
                html.Li("전하량을 2배로 바꾸면 같은 위치에서 |E|와 V는 어떻게 변할까?"),
            ]
        ),
    ],
    style={
        "maxWidth": "1250px",
        "margin": "0 auto",
        "padding": "18px",
        "fontFamily": "Arial, sans-serif",
    },
)


@app.callback(
    Output("field-graph", "figure"),
    Output("positions", "data"),
    Output("probe-readout", "children"),
    Output("position-readout", "children"),
    Input("field-graph", "relayoutData"),
    Input("q1", "value"),
    Input("q2", "value"),
    Input("probe-x", "value"),
    Input("probe-y", "value"),
    Input("display-options", "value"),
    Input("reset", "n_clicks"),
    State("positions", "data"),
)
def update_app(
    relayout_data,
    q1,
    q2,
    probe_x,
    probe_y,
    display_options,
    _reset_clicks,
    positions,
):
    positions = dict(positions or DEFAULT_POSITIONS)

    if ctx.triggered_id == "reset":
        positions = dict(DEFAULT_POSITIONS)
    elif ctx.triggered_id == "field-graph":
        positions = update_positions_from_relayout(relayout_data, positions)

    q1 = float(q1 if q1 is not None else 0.0)
    q2 = float(q2 if q2 is not None else 0.0)
    probe_x = float(probe_x if probe_x is not None else 0.0)
    probe_y = float(probe_y if probe_y is not None else 0.0)
    display_options = display_options or []

    charges = [
        (q1, positions["x1"], positions["y1"]),
        (q2, positions["x2"], positions["y2"]),
    ]

    ex, ey, emag, potential, theta = probe_measurement(probe_x, probe_y, charges)

    fig = build_figure(
        q1=q1,
        q2=q2,
        positions=positions,
        probe_x=probe_x,
        probe_y=probe_y,
        show_vectors="vectors" in display_options,
        show_potential="potential" in display_options,
    )

    probe_children = [
        html.Div(f"Eₓ = {ex:+.4f}"),
        html.Div(f"Eᵧ = {ey:+.4f}"),
        html.Div(f"|E| = {emag:.4f}"),
        html.Div(f"V = {potential:+.4f}"),
        html.Div(f"θ = {theta:+.2f}°"),
    ]

    position_text = (
        f"q1 position = ({positions['x1']:+.2f}, {positions['y1']:+.2f})    "
        f"q2 position = ({positions['x2']:+.2f}, {positions['y2']:+.2f})"
    )

    return fig, positions, probe_children, position_text


if __name__ == "__main__":
    app.run(debug=True)
