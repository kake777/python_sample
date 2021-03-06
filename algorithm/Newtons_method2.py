class Newton(object):
    def __init__(self, f, d1f, d2f, f2):
        self.f = f
        self.d1f = d1f
        self.d2f = d2f
        self.f2 = f2

    def _update(self, bar_x):
        d1f = self.d1f
        d2f = self.d2f
        return bar_x - d1f(bar_x)/d2f(bar_x)

    def solve(self, init_x, n_iter, tol):
        bar_x = init_x
        for i in range(n_iter):
            x = self._update(bar_x)
            error = abs(x - bar_x)
            print("|Δx| = {0:.2f}, x = {1:.2f}".format(error, x))
            bar_x = x
            if error < tol:
                break
        return x

def _main():
    f = lambda x: x**3 - 2*x**2 + x + 3
    d1f = lambda x: 3*x**2 - 4*x + 1
    d2f = lambda x: 6*x - 4
    f2 = lambda bar_x, x: f(bar_x) + d1f(bar_x)*(x - bar_x) + d2f(bar_x)*(x - bar_x)**2

    newton = Newton(f=f, d1f=d1f, d2f=d2f, f2=f2)
    res = newton.solve(init_x=10, n_iter=100, tol=0.01)
    print("Solution is {0:.2f}".format(res))

if __name__ == "__main__":
    _main()
