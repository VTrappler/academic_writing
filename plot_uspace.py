# -*- coding: utf-8 -*-
#!/usr/bin/env python


exec(open('/home/victor/acadwriting/Manuscrit/plots_settings.py').read())

def main(i=0):
    uv = [0, 0.5, 1]
    plt.figure(figsize = (3, 3))
    plt.xlabel(r'$u_1$')
    plt.ylabel(r'$u_2$')
    x, y = np.stack(np.meshgrid(uv, uv)).T.reshape(-1, 2)[i]
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])
    plt.scatter([0.5], [0.5])
    plt.scatter(x, y)
    plt.tight_layout()
    plt.show()



if __name__ == '__main__':
    main()
