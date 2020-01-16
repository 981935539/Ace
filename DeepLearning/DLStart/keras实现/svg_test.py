# import svgwrite
#
# dwg = svgwrite.Drawing('test.svg', profile='tiny')
# dwg.add(dwg.line((0, 0), (10, 0), stroke=svgwrite.rgb(10, 10, 16, '%')))
# dwg.add(dwg.text('123', insert=(0, 0.2), fill='red'))
# dwg.save()

# import matplotlib.pyplot as plt
#
# # 随意绘制一个样图
# plt.plot([1, 2, 3, 4, 3, 2, 3])
#
# # 保存图为svg格式，即矢量图格式
# plt.savefig("test.svg", format="svg")


from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt

# font0 = FontProperties()
# alignment = {'horizontalalignment': 'center', 'verticalalignment': 'baseline'}
# # Show family options
#
# families = ['serif', 'sans-serif', 'monospace']
#
# font1 = font0.copy()
# font1.set_size('large')
#
# # t = plt.figtext(0.1, 0.9, 'family', fontproperties=font1, **alignment)
#
# yp = [0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2]
# digits = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
#
# for k, digit in enumerate(digits, start=1):
#     # subplot = plt.subplot(5, 2, k + 1)
#
#     font = font0.copy()
#     font.set_family("serif")
#     t = plt.figtext((k % 5)*0.2, (k // 5)*0.2, digit, fontproperties=font1, **alignment)
#
# plt.savefig("test.svg", format="svg")


import matplotlib.patches as patches

# build a rectangle in axes coords
left, width = .25, .5
bottom, height = .25, .5
right = left + width
top = bottom + height

fig = plt.figure()
ax = fig.add_axes([0, 0, 1, 1])

# axes coordinates are 0,0 is bottom left and 1,1 is upper right
# p = patches.Rectangle(
#     (left, bottom), width, height,
#     fill=False, transform=ax.transAxes, clip_on=False
#     )
#
# ax.add_patch(p)

ax.text(0.5*(left+right), 0.5*(bottom+top), '123',
        horizontalalignment='center',
        verticalalignment='center',
        fontsize=20, color='red',
        transform=ax.transAxes)



ax.set_axis_off()
# plt.show()

plt.savefig("test.svg", format="svg")