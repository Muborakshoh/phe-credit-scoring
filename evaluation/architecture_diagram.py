def draw_architecture():
    """Строит архитектурную диаграмму агентного пайплайна.

    Требует установленного системного пакета graphviz.
    """
    from graphviz import Digraph

    g = Digraph(format="png")
    g.attr(rankdir="LR", fontsize="11")

    with g.subgraph(name="cluster_client") as c:
        c.attr(label="CLIENT", style="dashed", color="blue", fontsize="12")
        c.node("A", "Raw Features", shape="box")
        c.node("M", "MonitoringAgent\n(validate)", shape="box",
               style="filled", fillcolor="lightyellow")
        c.node("E", "EncryptionAgent\n(Paillier PK)", shape="box",
               style="filled", fillcolor="lightyellow")
        c.node("T", "TransmissionAgent\n(HMAC-SHA256)", shape="box",
               style="filled", fillcolor="lightyellow")
        c.node("D", "Decrypt z\n(Paillier SK)", shape="box")
        c.node("SIG", "Sigmoid / Approx", shape="box")
        c.node("OUT", "Credit Score", shape="ellipse",
               style="filled", fillcolor="lightgreen")

    with g.subgraph(name="cluster_server") as s:
        s.attr(label="SERVER", style="dashed", color="red", fontsize="12")
        s.node("AN", "AnalysisAgent\nΣ w_i·E(x_i)+E(b)", shape="box",
               style="filled", fillcolor="#ffe0e0")

    g.edge("A", "M", label="raw row")
    g.edge("M", "E", label="valid")
    g.edge("E", "T", label="E(x)")
    g.edge("T", "AN", label="E(x) + MAC")
    g.edge("AN", "D", label="E(z)")
    g.edge("D", "SIG", label="z")
    g.edge("SIG", "OUT")

    return g
