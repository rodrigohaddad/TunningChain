import json
import matplotlib.pyplot as plt

def main():
    file_name_title = [('sa_t1_03_250000', 'beta=0.3'), ('sa_t1_06_250000', 'beta=0.6'), ('sa_t1_09_250000', 'beta=0.9'),
                        ('sa_t1_03_10000', 'beta=0.3'), ('sa_t1_06_10000', 'beta=0.6'), ('sa_t1_09_10000', 'beta=0.9'),
                        ('sa_t2_03_2000', 'beta=0.3'), ('sa_t2_06_2000', 'beta=0.6'), ('sa_t2_09_2000', 'beta=0.9'),
                        ('sa_t3_20_10', 'a=20, b=10'), ('sa_t3_200_100', 'a=200, b=100'), ('sa_t3_2000_1000', 'a=2000, b=1000'),
                        ]
    for ff in file_name_title:
        with open(ff[0], 'r') as f:
            data = json.load(f)

            fig, ax = plt.subplots()
            if ff[0].startswith("sa_t1"):
                p1,=plt.plot(data[2][:500], '--r')
            else:
                p1,=plt.plot(data[2][:2000], '--r')
            plt.grid(True)
            ax.set_title(ff[1])
            ax.tick_params()

            # Get second axis
            ax2 = ax.twinx()
            if ff[0].startswith("sa_t1"):
                p2,=plt.plot(data[3][:500], 'cornflowerblue')
            else:
                p2,=plt.plot(data[3][:2000], 'cornflowerblue')
            ax.tick_params()  
            plt.legend([p1, p2], ['Temperatura', 'Acurácia'])

            plt.savefig(fname=f"plots/{ff[0]}_pic")

if __name__ == '__main__':
    main()