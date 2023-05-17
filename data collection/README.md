我们的实验使用hololens2对点云数据进行采集。我们参考和改进了的工作，在hololens2中部署了离线点云采集应用。

# 实验环境
visual studio 2022
hololens2
unity-可选，对项目进行改动再重新打包

此外，我们还需要做一些设置
## 打开研究者模式
我们将hololens和电脑同时连接一个wifi，将hololens中的ipv4地址输入到电脑的浏览器中，在隐私错误的网页中选择高级-继续访问 \
随后将设置好的用户名和密码填写在网页上方的登录小窗 \
关于此部分可以参考[微软官方hololens的介绍](https://learn.microsoft.com/zh-cn/hololens/hololens2-hardware)\
登录成功后可进入以下页面\
[图]
在左侧菜单中选择System-Research mode 勾选Allow access to sensor streams\
[图]

## 更新hololens的windows内部版本
只有预览体验成员可以部署Windows 11和Windows 10 Insider Preview 版本\
我们可以在hololens的设置-更新于安全-windows预览体验计划\
更多信息请参考[windos支持网站](https://support.microsoft.com/zh-cn/windows/%E5%8A%A0%E5%85%A5-windows-%E9%A2%84%E8%A7%88%E4%BD%93%E9%AA%8C%E8%AE%A1%E5%88%92%E5%B9%B6%E7%AE%A1%E7%90%86%E9%A2%84%E8%A7%88%E4%BD%93%E9%AA%8C%E6%88%90%E5%91%98%E8%AE%BE%E7%BD%AE-ef20bb3d-40f4-20cc-ba3c-a72c844b563c)获得可用的windows预览体验成员账户。
期间会接收到邮件，其中有部分视频说明以便更好的安装windows内部版本。
之后，我们可以在设置-windows更新中看到可用的windows版本，下载即可。
本次数据采集的过程中，我们的Windows version是22621.1057.arm64fre.ni release svc sydney.230222-1630
这一部分内容若有技术问题也可以寻求hololens客服帮助400-820-3800

# 配置流程
1.首先下载Microsoft HoloLens 2 计算机视觉研究的[示例代码和文档](https://github.com/microsoft/HoloLens2ForCV/tree/main)。
其中找到Samples/StreamRecorder文件，这是我们采集点云数据的主要示例文件
2.在Visual Studio的安装中需记得添加USB设备连接性，否则会出现DEP6957: 未能使用“通用身份验证”连接到设备“127.0.0.1”错误
3.在Visual Studio2022中打开StreamRecorder/StreamRecorderApp/StreamRecorder.sln文件，其属性页如下：[图]
4.通过usb连接hololens，选择Release-ARM64-设备 运行文件并等待部署完成即可退出。其中，hololens首次与电脑配对需要在Visual Studio中输入hololensd PIN码完成配对。[图]
以上步骤完成之后可以在hololens的应用中找到我们刚才部署的文件[图]

# 开始采集
打开hololens的xx项目，点击start开始采集数据，点击stop停止采集数据。该应用是离线的，使用者在hololens电量充足的地方可到任何室内场景进行采集
