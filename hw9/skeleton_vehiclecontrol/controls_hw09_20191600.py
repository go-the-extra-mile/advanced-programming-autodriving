import matplotlib.pyplot as plt

# initial and reference positions
x_0 = 0 # Starting position
x_ref = 10 # Destination position

# simulation parameters
delta_t = 0.1
T = 300
N = int(T / delta_t)

def bang_bang(x, x_ref):
    # Bang-bang control
    e = x_ref - x 
    if e >= 0:
        a = 1
    elif e < 0:
        a = -1
    else:
        a = 0
    return a

def P(x, x_ref, K_p):
    # P control
    e = x_ref - x
    a = K_p * e # Acceleration
    return a

def PD(x, x_ref, v, K_p, K_d):
    # PD control
    e = x_ref - x 
    de = -v # Derivative of the error. e(i+1) - e(i) = x(i) - x(i+1) = -v(i)
    a = K_p * e + K_d * de # Acceleration
    return a

def run_simulation(ctrl_method="bang bang", K_p=0.05, K_d=0.1):
    x_ref = [0] * 20 + [1] * (N - 20)

    # lists of positions for each control method
    x_bb = [x_0] # Bang-bang
    x_p = [x_0] # P
    x_pd = [x_0] # PD

    # lists of velocities for each control method
    v_bb = [0] # Bang-bang
    v_p = [0] # P
    v_pd = [0] # PD

    x = None
    if ctrl_method == "bang bang": 
        for i in range(N):
            # Update the positions and velocities using the explicit Euler method
            x_bb.append(x_bb[i] + v_bb[i] * delta_t)
            v_bb.append(v_bb[i] + bang_bang(x_bb[i], x_ref[i]) * delta_t)
        x = x_bb
    elif ctrl_method == "P":
        for i in range(N):
            x_p.append(x_p[i] + v_p[i] * delta_t)
            v_p.append(v_p[i] + P(x_p[i], x_ref[i], K_p) * delta_t)
        x = x_p
    elif ctrl_method == "PD":
        for i in range(N):
            x_pd.append(x_pd[i] + v_pd[i] * delta_t)
            v_pd.append(v_pd[i] + PD(x_pd[i], x_ref[i], v_pd[i], K_p, K_d) * delta_t)
        x = x_pd
    else:
        print("invalid control method provided")
        return

    # Plot the positions for each control method
    plt.xlabel("t")
    plt.plot([i * delta_t for i in range(N)], x[:N], color="red", linestyle="--", label="y(t)")
    plt.plot([i * delta_t for i in range(N)], x_ref, color="blue", label="r(t)")
    
    plot_title = str()
    if ctrl_method == "bang bang":
        plot_title = "Bang-Bang"
    elif ctrl_method == "P":
        plot_title = f"{ctrl_method} (K_p={K_p})"
    elif ctrl_method == "PD":
        plot_title = f"{ctrl_method} (K_p={K_p} K_d={K_d})"
    plt.title(plot_title)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    experiments = [
        ("P", 0.05, None),
        # ("P", 0.5, None),
        # ("PD", 0.05, 0.1),
        # ("PD", 0.05, 0.5),
        # ("bang bang", None, None),
    ]

    for e in experiments:
        run_simulation(*e)