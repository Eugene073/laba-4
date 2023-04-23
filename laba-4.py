'''
29.	Формируется матрица F следующим образом: скопировать в нее А и  если в С количество нулей в нечетных столбцах больше,
чем сумма чисел по периметру , то поменять местами С и В симметрично, иначе С и Е поменять местами несимметрично.
При этом матрица А не меняется. После чего если определитель матрицы А больше суммы диагональных элементов матрицы F,
то вычисляется выражение:A-1*AT – K * F-1, иначе вычисляется выражение (A +G-1-F-1)*K, где G-нижняя треугольная матрица,
полученная из А. Выводятся по мере формирования А, F и все матричные операции последовательно
'''

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

print("Введите число для задания размерности квадратной матрицы(не менее 4)")
N = int(input())
print("Введите число <K> ")
K = int(input())
if N<4:
    print("Указана слишком маленькая размерность матрицы")
    exit(0)
matrixA = np.random.randint(-10, 10, (int(N), int(N)))
matrixF = matrixA.copy()
l1 = N//2
np.set_printoptions(linewidth=1000)

print("Матрица А\n", matrixA, "\n", "-------------------")
print("Начальная матрица F\n", matrixF, "\n", "-------------------")
if N % 2 == 0:
    matB = matrixA[0:l1, 0:l1]
    print("Матрица B\n", matB, "\n", "-------------------")
    matC = matrixA[0:l1, l1:]
    print("Матрица C\n", matC, "\n", "-------------------")
    matD = matrixA[l1:, 0:l1]
    print("Матрица D\n", matD, "\n", "-------------------")
    matE = matrixA[l1:, l1:]
    print("Матрица E\n", matE, "\n", "-------------------")
else:
    matB = matrixA[0:l1, 0:l1]
    print("Матрица B\n", matB, "\n", "-------------------")
    matC = matrixA[0:l1, l1+1:]
    print("Матрица C\n", matC, "\n", "-------------------")
    matD = matrixA[l1+1:, 0:l1]
    print("Матрица D\n", matD, "\n", "-------------------")
    matE = matrixA[l1+1:, l1+1:]
    print("Матрица E\n", matE, "\n", "-------------------")


ch = []  # Нули по нечетным столбцам
summ = 0  # сумма чисел по пириметру


for i in range(0, l1, 2): # счетчик нулей
    for j in range(0, l1, 1):
        ch.append(matC[j][i])
zero = 0
for i in range(len(ch)):
    if int(ch[i]) == 0:
        zero += 1
print("Количество нулей в нечетных столбцах подматрицы С =", zero, "\n", "-------------------")


for i in range(0, l1, 1):  # сумма по пириметру
    summ = summ + matC[i][0] + matC[i][-1]
for i in range(1, l1-1, 1):
    summ = summ + matC[0][i] + matC[-1][i]
print("Сумма чисел по пириметру подматрицы С =", summ, "\n", "-------------------")

if zero > summ:
    print("Количество нулей в нечетных столбцах подматрицы C больше, чем сумма чисел по пириметру подматрицы С:", zero, ">", summ, ". "
    " Меняем симметрично B и C.", "\n", "-------------------")
    matrixF[:l1, l1+N % 2:N] = matB[:l1, ::-1]
    matrixF[:l1, :l1] = matC[:l1, ::-1]
else:
    print("Количество нулей в нечетных столбцах подматрицы C меньше или равно, чем сумма чисел по пириметру подматрицы С:", zero, "<=", summ, "."
    " Меняем несимметрично B и E.", "\n", "-------------------")
    matrixF[:l1, :l1] = matE
    matrixF[l1+N % 2:N, l1+N % 2:N] = matB
print("Измененная матрица F\n", matrixF, "\n", "-------------------")


try:
    if np.linalg.det(matrixA) > sum(np.diagonal(matrixF)):
        print("\nРезультат выражения A^(-1) *AT – K * F^(-1):\n", np.linalg.inv(matrixA) * matrixA.transpose() - K * np.linalg.inv(matrixF))
    else:
        G = np.tri(N) * matrixA
        print("\nРезультат выражения (A + G^(-1) - F^(-1) )*K:\n", (matrixA + np.linalg.inv(G) - np.linalg.inv(matrixF)) * K)

except np.linalg.LinAlgError:
    print("Обратную матрицу найти невозможно т.к. определитель равен 0.")

av = [np.mean(abs(matrixF[i, ::])) for i in range(N)]
av = int(sum(av))                                       # сумма средних значений строк (используется при создании третьего графика)
fig, axs = plt.subplots(2, 2, figsize=(11, 8))
x = list(range(1, N+1))
for j in range(N):
    y = list(matrixF[j, ::])                                      # обычный график
    axs[0, 0].plot(x, y, ',-', label=f"{j+1} строка.")
    axs[0, 0].set(title="График с использованием функции plot:", xlabel='Номер элемента в строке', ylabel='Значение элемента')
    axs[0, 0].grid()
    axs[0, 1].bar(x, y, 0.4, label=f"{j+1} строка.")                # гистограмма
    axs[0, 1].set(title="График с использованием функции bar:", xlabel='Номер элемента в строке', ylabel='Значение элемента')
    if N <= 10:
        axs[0, 1].legend(loc='lower right')
        axs[0, 1].legend(loc='lower right')
explode = [0]*(N-1)                                     # отношение средних значений от каждой строки
explode.append(0.1)
sizes = [round(np.mean(abs(matrixF[i, ::])) * 100/av, 1) for i in range(N)]
axs[1, 0].set_title("График с ипользованием функции pie:")
axs[1, 0].pie(sizes, labels=list(range(1, N+1)), explode=explode, autopct='%1.1f%%', shadow=True)
def heatmap(data, row_labels, col_labels, ax, cbar_kw={}, **kwargs):            # аннотированная тепловая карта
    im = ax.imshow(data, **kwargs)
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    ax.set_xticks(np.arange(data.shape[1]), labels=col_labels)
    ax.set_yticks(np.arange(data.shape[0]), labels=row_labels)
    return im, cbar
def annotate_heatmap(im, data = None, textcolors=("black","white"), threshold=0):
    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()
    kw = dict(horizontalalignment="center", verticalalignment="center")
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(data[i, j] > threshold)])
            text = im.axes.text(j, i, data[i, j], **kw)
            texts.append(text)
    return texts
im, cbar = heatmap(matrixF, list(range(N)), list(range(N)), ax=axs[1, 1], cmap="magma_r")
texts = annotate_heatmap(im)
axs[1, 1].set(title="Создание аннотированных тепловых карт:", xlabel="Номер столбца", ylabel="Номер строки")
plt.suptitle("Использование библиотеки matplotlib")
plt.tight_layout()
plt.show()
    # использование библиотеки seaborn
number_row = []
for i in range(1, N+1):
    number_row += [i]*N
number_item = list(range(1, N+1))*N
df = pd.DataFrame({"Значения": matrixF.flatten(), "Номер строки": number_row, "Номер элемента в строке": number_item})
fig, axs = plt.subplots(2, 2, figsize=(11, 8))
plt.subplot(221)
plt.title("Использование функции lineplot")
sns.lineplot(x="Номер элемента в строке", y="Значения", hue="Номер строки", data=df, palette="Set2")
plt.subplot(222)
plt.title("Использование функции boxplot")
sns.boxplot(x="Номер строки", y="Значения", palette="Set2", data=df)
plt.subplot(223)
plt.title("Использование функции kdeplot")
sns.kdeplot(data=df, x="Номер элемента в строке", y="Значения", hue="Номер строки", palette="Set2")
plt.subplot(224)
plt.title("Использование функции heatmap")
sns.heatmap(data=matrixF, annot=True, fmt="d", linewidths=.5)
plt.suptitle("Использование библиотеки seaborn")
plt.tight_layout()
plt.show()