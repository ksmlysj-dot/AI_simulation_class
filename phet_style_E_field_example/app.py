from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, ctx, dcc, html

from physics import acceleration, potential, potential_energy, step_state, time_to_land


# ------------------------------------------------------------
# Classroom defaults
# ------------------------------------------------------------
D_DEFAULT = 4.0
E_DEFAULT = 1.0
Q_DEFAULT = 1.0
M_DEFAULT = 1.0
Y0_DEFAULT = D_DEFAULT - 0.3

PLATE_HALF_WIDTH = 3.5
CHARGE_RADIUS = 0.18
INTERVAL_MS = 50

# Fixed velocity-graph frame: the axis range and tick spacing never change
# with q/E/m/y0, so a steeper or shallower line is the only visible sign
# that the acceleration changed - the grid itself never moves.
V_GRAPH_T_MAX = 8.0
V_GRAPH_T_DTICK = 1.0
V_GRAPH_V_MAX = 10.0
V_GRAPH_V_DTICK = 1.0

DEFAULT_SIM_STATE = {
    "y": Y0_DEFAULT,
    "v": 0.0,
    "t": 0.0,
    "running": False,
    "history": [[0.0, 0.0]],
}


def fresh_sim_state(y0: float) -> dict:
    return {"y": y0, "v": 0.0, "t": 0.0, "running": False, "history": [[0.0, 0.0]]}


def build_figure(
    d: float,
    E: float,
    q: float,
    y_current: float,
    show_vectors: bool,
    show_potential: bool,
) -> go.Figure:
    fig = go.Figure()

    # Equipotential lines (V depends only on y, so these are horizontal)
    if show_potential:
        for frac in (0.25, 0.5, 0.75):
            y_line = frac * d
            v_line = potential(y_line, E, d)
            fig.add_shape(
                type="line",
                x0=-PLATE_HALF_WIDTH,
                x1=PLATE_HALF_WIDTH,
                y0=y_line,
                y1=y_line,
                line=dict(color="rgba(120,120,120,0.6)", width=1, dash="dot"),
            )
            fig.add_annotation(
                x=PLATE_HALF_WIDTH + 0.15,
                y=y_line,
                text=f"V={v_line:.2f}",
                showarrow=False,
                xanchor="left",
                font=dict(size=10, color="gray"),
            )

    # Uniform, downward electric-field arrows between the plates
    if show_vectors:
        cols = np.linspace(-PLATE_HALF_WIDTH + 0.6, PLATE_HALF_WIDTH - 0.6, 7)
        rows = np.linspace(d * 0.15, d * 0.85, 4)
        arrow_len = min(0.35, d * 0.15)

        xs: list[float | None] = []
        ys: list[float | None] = []
        hx: list[float] = []
        hy: list[float] = []
        for x0 in cols:
            for y0_row in rows:
                y_end = y0_row - arrow_len
                xs.extend([x0, x0, None])
                ys.extend([y0_row, y_end, None])
                hx.append(x0)
                hy.append(y_end)

        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="lines",
                line=dict(width=1, color="rgba(30,90,190,0.8)"),
                hoverinfo="skip",
                name="E field",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=hx,
                y=hy,
                mode="markers",
                marker=dict(symbol="triangle-down", size=7, color="rgba(30,90,190,0.8)"),
                hoverinfo="skip",
                showlegend=False,
            )
        )

    # Top (+) and bottom (-) plates
    fig.add_shape(
        type="rect",
        x0=-PLATE_HALF_WIDTH,
        x1=PLATE_HALF_WIDTH,
        y0=d,
        y1=d + 0.18,
        fillcolor="rgba(190,60,60,0.85)",
        line=dict(width=0),
    )
    fig.add_shape(
        type="rect",
        x0=-PLATE_HALF_WIDTH,
        x1=PLATE_HALF_WIDTH,
        y0=-0.18,
        y1=0,
        fillcolor="rgba(60,90,190,0.85)",
        line=dict(width=0),
    )
    for xp in np.linspace(-PLATE_HALF_WIDTH + 0.3, PLATE_HALF_WIDTH - 0.3, 9):
        fig.add_annotation(x=xp, y=d + 0.09, text="+", showarrow=False, font=dict(size=16, color="white"))
        fig.add_annotation(x=xp, y=-0.09, text="−", showarrow=False, font=dict(size=16, color="white"))

    # Reference line/label at the bottom plate (PE = 0)
    fig.add_annotation(
        x=-PLATE_HALF_WIDTH + 0.15,
        y=0,
        text="기준 (위치 에너지: 0)",
        showarrow=False,
        xanchor="left",
        yanchor="bottom",
        font=dict(size=11),
    )

    # Dashed vertical drop line + horizontal PE line from the charge
    pe_current = potential_energy(y_current, q, E, d)
    fig.add_shape(
        type="line",
        x0=0,
        x1=0,
        y0=0,
        y1=y_current,
        line=dict(color="rgba(0,0,0,0.5)", width=1, dash="dash"),
    )
    fig.add_shape(
        type="line",
        x0=-PLATE_HALF_WIDTH + 1.6,
        x1=0,
        y0=y_current,
        y1=y_current,
        line=dict(color="rgba(0,0,0,0.4)", width=1, dash="dash"),
    )
    fig.add_annotation(
        x=-PLATE_HALF_WIDTH + 0.15,
        y=y_current,
        text=f"위치 에너지: qEy ≈ {pe_current:.2f}",
        showarrow=False,
        xanchor="left",
        yanchor="bottom",
        font=dict(size=11),
    )

    # Force vector qE, pointing down from the charge
    force = q * E
    arrow_len = min(0.9, 0.25 * force + 0.2)
    fig.add_annotation(
        x=0.55,
        y=max(y_current - arrow_len, 0.0),
        ax=0.55,
        ay=y_current,
        xref="x",
        yref="y",
        axref="x",
        ayref="y",
        showarrow=True,
        arrowhead=3,
        arrowwidth=2,
        arrowcolor="rgba(220,90,40,0.9)",
    )
    fig.add_annotation(
        x=0.85,
        y=max(y_current - arrow_len / 2, 0.0),
        text="qE",
        showarrow=False,
        font=dict(size=12, color="rgba(220,90,40,0.9)"),
    )

    # The moving unit charge
    fig.add_shape(
        type="circle",
        x0=-CHARGE_RADIUS,
        x1=CHARGE_RADIUS,
        y0=y_current - CHARGE_RADIUS,
        y1=y_current + CHARGE_RADIUS,
        fillcolor="rgba(150,60,170,0.9)",
        line=dict(color="rgba(90,20,110,1)", width=2),
    )
    fig.add_annotation(
        x=0,
        y=y_current + 0.35,
        text=f"+q ({q:.1f})",
        showarrow=False,
        font=dict(size=12),
    )

    fig.update_layout(
        height=650,
        margin=dict(l=25, r=90, t=20, b=20),
        xaxis=dict(
            range=[-PLATE_HALF_WIDTH - 0.5, PLATE_HALF_WIDTH + 0.5],
            visible=False,
            fixedrange=True,
            scaleanchor="y",
            scaleratio=1,
        ),
        yaxis=dict(
            range=[-0.6, d + 0.9],
            title="높이 y",
            fixedrange=True,
            zeroline=False,
        ),
        showlegend=False,
        uirevision="capacitor-classroom",
    )
    return fig


def build_velocity_figure(
    history: list[list[float]],
    y_current: float,
    v_current: float,
    t_current: float,
    a: float,
) -> go.Figure:
    """Plot the actual v(t) travelled so far plus a live forward projection.

    `history` holds every (t, v) sample recorded tick by tick, so if q, E or
    m changed mid-fall the recorded slope visibly kinks at that instant
    instead of being a single straight line. The dotted projection always
    extrapolates from the charge's *current* state using the *current*
    acceleration, so its slope updates the moment a slider changes.

    The axis range and tick spacing (V_GRAPH_*) are fixed constants, never
    recomputed from the trajectory - so when q, E or m change, the grid
    itself stays put and only the line's slope visibly changes.
    """
    fig = go.Figure()

    t_hist = [point[0] for point in history]
    v_hist = [point[1] for point in history]

    t_land = time_to_land(y_current, v_current, a) if y_current > 1e-9 else 0.0
    if t_land > 0:
        t_pred_full = np.linspace(t_current, t_current + t_land, 30)
        v_pred_full = v_current + a * (t_pred_full - t_current)
        in_frame = (t_pred_full <= V_GRAPH_T_MAX) & (v_pred_full <= V_GRAPH_V_MAX)
        visible_count = int(np.argmin(in_frame)) if not in_frame.all() else len(in_frame)
        t_pred = t_pred_full[:visible_count]
        v_pred = v_pred_full[:visible_count]
    else:
        t_pred = np.array([])
        v_pred = np.array([])

    if len(t_pred) > 1:
        fig.add_trace(
            go.Scatter(
                x=t_pred,
                y=v_pred,
                mode="lines",
                line=dict(color="rgba(120,120,120,0.5)", width=1.5, dash="dot"),
                hoverinfo="skip",
                name="지금 기울기 유지 시 예상",
            )
        )
        if len(t_pred) == 30:  # the full projection to landing fit on screen
            fig.add_annotation(
                x=t_pred[-1],
                y=v_pred[-1],
                text=f"착지 예상 v ≈ {v_pred[-1]:.2f}",
                showarrow=True,
                arrowhead=2,
                ax=-35,
                ay=-20,
                font=dict(size=10),
            )

    fig.add_trace(
        go.Scatter(
            x=t_hist,
            y=v_hist,
            mode="lines",
            line=dict(color="rgba(150,60,170,0.9)", width=3),
            hoverinfo="skip",
            name="v(t)",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[t_current],
            y=[v_current],
            mode="markers",
            marker=dict(color="rgba(150,60,170,1)", size=10),
            hoverinfo="skip",
            showlegend=False,
        )
    )

    fig.update_layout(
        height=650,
        margin=dict(l=50, r=15, t=20, b=40),
        xaxis=dict(
            title="시간 t (s)",
            range=[0, V_GRAPH_T_MAX],
            dtick=V_GRAPH_T_DTICK,
            autorange=False,
            fixedrange=True,
            zeroline=True,
        ),
        yaxis=dict(
            title="속도 v",
            range=[0, V_GRAPH_V_MAX],
            dtick=V_GRAPH_V_DTICK,
            autorange=False,
            fixedrange=True,
            zeroline=True,
        ),
        showlegend=False,
        uirevision="velocity-graph",
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
        dcc.Store(id="sim-state", data=dict(DEFAULT_SIM_STATE)),
        dcc.Interval(id="ticker", interval=INTERVAL_MS, n_intervals=0, disabled=True),

        html.H1("⚡ Electric Field Playground", style={"marginBottom": "4px"}),
        html.P(
            "평행판 축전기 사이의 균일한 전기장 안에 양전하를 놓고, "
            "그 전하가 아래로 떨어지는 모습을 관찰해 보세요.",
            style={"marginTop": "0"},
        ),

        html.Div(
            [
                # Left controls
                html.Div(
                    [
                        html.Div(
                            [
                                html.H3("판 & 전기장"),
                                html.Label("판 간격 d"),
                                dcc.Slider(
                                    id="d-slider",
                                    min=2.0,
                                    max=6.0,
                                    step=0.5,
                                    value=D_DEFAULT,
                                    marks={2: "2", 4: "4", 6: "6"},
                                ),
                                html.Br(),
                                html.Label("전기장 세기 E"),
                                dcc.Slider(
                                    id="e-slider",
                                    min=0.2,
                                    max=2.0,
                                    step=0.1,
                                    value=E_DEFAULT,
                                    marks={0.2: "0.2", 1: "1", 2: "2"},
                                ),
                            ],
                            style=CARD_STYLE,
                        ),

                        html.Div(
                            [
                                html.H3("전하 설정"),
                                html.Label("전하량 q (+)"),
                                dcc.Slider(
                                    id="q-slider",
                                    min=0.2,
                                    max=3.0,
                                    step=0.1,
                                    value=Q_DEFAULT,
                                    marks={0.2: "0.2", 1: "1", 3: "3"},
                                ),
                                html.Br(),
                                html.Label("질량 m"),
                                dcc.Slider(
                                    id="m-slider",
                                    min=0.2,
                                    max=3.0,
                                    step=0.1,
                                    value=M_DEFAULT,
                                    marks={0.2: "0.2", 1: "1", 3: "3"},
                                ),
                                html.Br(),
                                html.Label("초기 높이 y₀"),
                                dcc.Slider(
                                    id="y0-slider",
                                    min=0.0,
                                    max=D_DEFAULT,
                                    step=0.1,
                                    value=Y0_DEFAULT,
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
                                html.Button(
                                    "놓기 (낙하 시작)",
                                    id="release",
                                    n_clicks=0,
                                    style={"padding": "9px 14px", "cursor": "pointer", "marginRight": "8px"},
                                ),
                                html.Button(
                                    "리셋",
                                    id="reset",
                                    n_clicks=0,
                                    style={"padding": "9px 14px", "cursor": "pointer"},
                                ),
                            ]
                        ),

                        html.Div(id="sim-readout", style={"marginTop": "14px", "fontFamily": "monospace"}),
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
                    ],
                    style={"flex": "1 1 700px", "minWidth": "0"},
                ),

                # Velocity-time graph
                html.Div(
                    [
                        html.H3("시간-속도 그래프", style={"marginTop": "0"}),
                        dcc.Graph(
                            id="velocity-graph",
                            config={
                                "displaylogo": False,
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
                    ],
                    style={"flex": "0 0 340px", "minWidth": "280px"},
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
                html.Li("초기 높이 y₀를 높이면 바닥에 닿는 속도는 어떻게 달라질까?"),
                html.Li("전하량 q를 2배로 늘리면 가속도 a=qE/m은 어떻게 변할까?"),
                html.Li("질량 m을 키우면 낙하 속도는 어떻게 변할까?"),
                html.Li("전하가 절반 높이에 있을 때 위치 에너지는 최대값의 몇 %일까?"),
            ]
        ),
    ],
    style={
        "maxWidth": "1650px",
        "margin": "0 auto",
        "padding": "18px",
        "fontFamily": "Arial, sans-serif",
    },
)


@app.callback(
    Output("y0-slider", "max"),
    Output("y0-slider", "value"),
    Input("d-slider", "value"),
    State("y0-slider", "value"),
)
def clamp_initial_height(d_value, y0_value):
    d_value = float(d_value)
    y0_value = d_value if y0_value is None else min(float(y0_value), d_value)
    return d_value, y0_value


@app.callback(
    Output("field-graph", "figure"),
    Output("velocity-graph", "figure"),
    Output("sim-state", "data"),
    Output("ticker", "disabled"),
    Output("sim-readout", "children"),
    Input("ticker", "n_intervals"),
    Input("release", "n_clicks"),
    Input("reset", "n_clicks"),
    Input("d-slider", "value"),
    Input("e-slider", "value"),
    Input("q-slider", "value"),
    Input("m-slider", "value"),
    Input("y0-slider", "value"),
    Input("display-options", "value"),
    State("sim-state", "data"),
)
def update_app(
    n_intervals,
    _release_clicks,
    _reset_clicks,
    d,
    E,
    q,
    m,
    y0,
    display_options,
    sim_state,
):
    d = float(d)
    E = float(E)
    q = float(q)
    m = float(m)
    y0 = float(y0)
    display_options = display_options or []
    sim_state = dict(sim_state or DEFAULT_SIM_STATE)

    triggered = ctx.triggered_id
    a = acceleration(q, E, m)

    if triggered == "reset":
        sim_state = fresh_sim_state(y0)
        ticker_disabled = True
    elif triggered == "release":
        sim_state = fresh_sim_state(y0)
        sim_state["running"] = True
        ticker_disabled = False
    elif triggered == "ticker" and sim_state.get("running"):
        dt = INTERVAL_MS / 1000.0
        y, v, dt_used, landed = step_state(sim_state["y"], sim_state["v"], a, dt)
        t = sim_state.get("t", 0.0) + dt_used
        history = sim_state.get("history", [[0.0, 0.0]])
        history.append([t, v])
        sim_state["y"] = y
        sim_state["v"] = v
        sim_state["t"] = t
        sim_state["history"] = history
        if landed:
            sim_state["running"] = False
        ticker_disabled = not sim_state["running"]
    elif triggered in ("q-slider", "e-slider", "m-slider") and sim_state.get("running"):
        # Mid-fall parameter change: keep flying from the current (y, v, t) -
        # the freshly recomputed `a` above takes effect starting next tick,
        # so the charge and the velocity graph's slope update in lockstep.
        ticker_disabled = False
    elif triggered in ("d-slider", "e-slider", "q-slider", "m-slider", "y0-slider"):
        sim_state = fresh_sim_state(y0)
        ticker_disabled = True
    else:
        ticker_disabled = not sim_state.get("running", False)

    y_current = sim_state.get("y", y0)
    v_current = sim_state.get("v", 0.0)
    t_current = sim_state.get("t", 0.0)
    history = sim_state.get("history", [[0.0, 0.0]])

    fig = build_figure(
        d=d,
        E=E,
        q=q,
        y_current=y_current,
        show_vectors="vectors" in display_options,
        show_potential="potential" in display_options,
    )
    fig_v = build_velocity_figure(
        history=history,
        y_current=y_current,
        v_current=v_current,
        t_current=t_current,
        a=a,
    )

    pe_current = potential_energy(y_current, q, E, d)
    pe_max = potential_energy(d, q, E, d)
    readout = [
        html.Div(f"a = qE/m = {a:.3f}"),
        html.Div(f"t = {t_current:.3f} s"),
        html.Div(f"y = {y_current:.3f}"),
        html.Div(f"v = {v_current:.3f}"),
        html.Div(f"PE = qEy = {pe_current:.3f} (최대 qEd = {pe_max:.3f})"),
    ]

    return fig, fig_v, sim_state, ticker_disabled, readout


if __name__ == "__main__":
    app.run(debug=True)
