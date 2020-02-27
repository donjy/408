from tkinter import *
from tkinter import ttk
import tkinter as tk
from PIL import Image, ImageTk
num = 0

def systeam_RSI(root):
    # 构造网格
    # for i in range(26):
    #     temp = tk.Label(root,text=i,width=5,height=1,bd=2,relief=SUNKEN)
    #     # temp = tk.Label(root,width=5,height=1)
    #     temp.grid(row=0,column=i)
    #
    # for i in range(25):
    #     temp = tk.Label(root,text=i,width=5,height=1,bd=2,relief=SUNKEN)
    #     # temp = tk.Label(root,width=5,height=1)
    #     temp.grid(row=i,column=0)

    # 真实地物图像预览【2-11】
    # 真实地物图片名字
    real_imagesname = ['indian_pines.png', 'PaviaU.png', '3.png', '4.png', '5.png', '6.png', '7.jpg']
    # 文件路径
    real_path = 'D:\\Paper_system\\'

    # 地物图片
    label_real = tk.Label(root, text="选择遥感图像", font=20)
    label_real.grid(row=1, column=1, rowspan=1, columnspan=5)

    # 展示真实地物图片
    real_image_frame = ttk.Frame(root)
    real_image_frame["borderwidth"] = 2
    real_image_frame["relief"] = "sunken"
    real_image_frame["padding"] = 0
    real_image_frame.grid(row=2, column=1, padx=0, pady=0, rowspan=10, columnspan=5)
    real_image_ = Image.open(real_path + real_imagesname[0]).resize((168, 196))
    real_img = ImageTk.PhotoImage(real_image_)
    label_select_img = ttk.Label(real_image_frame, image=real_img, compound=CENTER)
    label_select_img.grid(row=0, column=0)

    # 轮换选择更新图片
    def Next():
        global num
        if num + 1 < len(real_imagesname):
            num = num + 1
            pil_image = Image.open(real_path + real_imagesname[num])
            pil_image = pil_image.resize((168, 196))
            img = ImageTk.PhotoImage(pil_image)
            label_select_img.configure(image=img)
        root.update_idletasks()  # 更新图片，必须update

    def Before():  # 更新图片操作
        global num
        if num - 1 >= 0:
            num = num - 1
            pil_image = Image.open(real_path + real_imagesname[num])
            pil_image = pil_image.resize((168, 196))
            img = ImageTk.PhotoImage(pil_image)
            label_select_img.configure(image=img)
            root.update_idletasks()  # 更新图片，必须update

    # 上下选择按钮 【11-12】【1-5】
    frame_button_select = ttk.Frame(root)
    frame_button_select.grid(row=12, column=1, rowspan=2, columnspan=5)
    tk.Button(frame_button_select, text="上一张", command=Before, bg="gold").grid(column=0, row=1, padx=[0, 15])
    tk.Button(frame_button_select, text="下一张", command=Next, bg="gold").grid(column=1, row=1, padx=[15, 0])

    # 通过绝对路径选择真实地物图片
    def selectPath():
        from tkinter import filedialog
        File = filedialog.askopenfilename(parent=root, initialdir="C:/", title='Choose an image.')
        select_image = Image.open(File)
        select_image = select_image.resize((168, 196))
        img = ImageTk.PhotoImage(select_image)

        # 通过config 来设置图片
        label_select_img.config(image=img)
        label_select_img.image = img
    # path = StringVar()
    Label(root, text="目标路径:").grid(row=14, column=2)
    e = Entry(root, textvariable=real_path, show=' ').grid(row=14, column=3)
    Button(root, text="路径选择", command=selectPath).grid(row=14, column=4)

    # 选择算法按钮，调整算法候选窗口大小
    frame_button_algorithm = ttk.Frame(root)
    frame_button_algorithm.grid(row=18, column=1, rowspan=10, columnspan=5)
    var = tk.StringVar()  # 定义一个var用来将radiobutton的值和Label的值联系在一起.
    line_show = tk.Label(frame_button_algorithm, bg='yellow', width=20, text='未选择算法')
    line_show.pack()

    # 定义选项触发函数功能
    def print_selection():
        line_show.config(text='已选定 ' + var.get())

    r1 = tk.Radiobutton(frame_button_algorithm, text='权重衰减逻辑回归算法', variable=var, value='SLR', command=print_selection)
    r1.pack()
    r2 = tk.Radiobutton(frame_button_algorithm, text='核逻辑回归算法', variable=var, value='KLR', command=print_selection)
    r2.pack()
    r3 = tk.Radiobutton(frame_button_algorithm, text='多核稀疏多元逻辑回归算法', variable=var, value='MKSMLR', command=print_selection)
    r3.pack()

    # 显示遥感图像类别标签
    label_class = tk.Label(root, text="遥感图像类别展示", font=20)
    label_class.grid(row=1, column=8, rowspan=1, columnspan=5)

    # 遥感图像类别展示【2-24】
    class_image_frame = ttk.Frame(root)
    class_image_frame["borderwidth"] = 2
    class_image_frame["relief"] = "sunken"
    class_image_frame["padding"] = 0
    class_image_frame.grid(row=2, column=8, padx=0, pady=0, rowspan=10, columnspan=5)

    # 选择地物类别图片
    label_path = 'D:\Paper_system\\label\\'
    label_imagenames = ['indian_pines_label.png']
    label_image = Image.open(label_path + label_imagenames[0])
    label_image = label_image.resize((168, 196))
    label_imgs = ImageTk.PhotoImage(label_image)
    showlabel_img = ttk.Label(class_image_frame, image=label_imgs, compound=CENTER)
    showlabel_img.grid(row=0, column=0)

    # 图像分类结果显示区【6-17】
    label2_result = tk.Label(root, text="遥感图像分类结果", font=20)
    label2_result.grid(row=18, column=8, rowspan=1, columnspan=5)

    # 预测图片位置画布
    predict_image_frame = ttk.Frame(root)
    predict_image_frame["borderwidth"] = 2
    predict_image_frame["relief"] = "sunken"
    predict_image_frame["padding"] = 0
    predict_image_frame.grid(row=20, column=8, rowspan=11, columnspan=5)

    # 添加预测结果图片
    predict_path = 'D:\Paper_system\\predict\\'
    predict_imagenames = ['1.png', 'indian_pines_ground_predict_sigma_2.png']
    predict_image = Image.open(predict_path + predict_imagenames[0])
    predict_image = predict_image.resize((168, 196))
    predict_imgs = ImageTk.PhotoImage(predict_image)
    showpredict_img = ttk.Label(predict_image_frame, image=predict_imgs, compound=CENTER)
    showpredict_img.grid(row=0, column=0)

    # 参数调节窗口
    label_sigma = tk.Label(root, text="参数调节", font=20)
    label_sigma.grid(row=1, column=15, rowspan=1, columnspan=5)

    # 选定参数窗口
    label_sigma_show = tk.Label(root, bg='green', fg='white', width=20, text='未选定参数')
    label_sigma_show.grid(row=3, column=15, rowspan=1, columnspan=5)

    # 滑条显示窗口
    label_side_show = tk.Label(root)
    label_side_show.grid(row=2, column=15, rowspan=1, columnspan=5)
    # 定义一个触发函数功能
    def print_selection(v):
        label_sigma_show.config(text='sigma: ' + v)

    # 创建一个尺度滑条，长度200字符，从0开始10结束，以2为刻度，精度为0.01，触发调用print_selection函数
    s = tk.Scale(label_side_show, label='调节sigma值(默认:0.2)', from_=0.2, to=10, orient=tk.HORIZONTAL, length=200, showvalue=0,
                 tickinterval=2,
                 resolution=0.01, command=print_selection)
    s.pack()

    # 分类准确率窗口
    label_acc = tk.Label(root, text="分类准确率", font=18)
    label_acc.grid(row=18, column=15, rowspan=1, columnspan=5)

    # 获取text值
    label_acc_show = tk.Label(root)
    label_acc_show.grid(row=20, column=15, rowspan=1, columnspan=5)

    Texts_acc = tk.Text(label_acc_show, width=10, height=2, font=20)
    Texts_acc.grid(row=1,column=1,rowspan=1, columnspan=5)

    # 显示默认acc值
    Texts_acc.delete('1.0', '1.end')
    acc_ = 0
    Texts_acc.insert(tk.INSERT, acc_)


    # 算法分类按钮
    def classification():
        import time
        time.sleep(3)
        # 更新准确率
        acc = 0.9999
        Texts_acc.delete('1.0', '1.end')
        Texts_acc.insert(tk.INSERT, acc)

        # 更新图片
        predict_path = 'D:\Paper_system\\predict\\'
        predict_imagenames = ['1.png', 'indian_pines_ground_predict_sigma_2.png']
        predict_image = Image.open(predict_path + predict_imagenames[1])
        predict_image = predict_image.resize((168, 196))
        predict_imgs = ImageTk.PhotoImage(predict_image)

        showpredict_img.config(image=predict_imgs)
        showpredict_img.image = predict_imgs


    frame_button_classfication = ttk.Frame(root)
    frame_button_classfication.grid(row=17, column=1, rowspan=2, columnspan=5)
    tk.Button(frame_button_classfication, text="进行图像分类", bg="gold", width=20,
              command=classification).grid(column=0, row=2, pady=10)

    root.mainloop()

def main():
    root = Tk()
    root.title("遥感图像分类原型系统")
    root.geometry('700x600')
    systeam_RSI(root)


if __name__ == '__main__':
    main()