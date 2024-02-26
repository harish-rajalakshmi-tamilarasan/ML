from pywinauto import Application

installer_path = r"C:\\Jenkins_slave\\workspace\\harish-4072-JOB\\AdventNetTestSuite\\SecureGatewayServer_64bit.exe"
app = Application().start(installer_path)
app.ManageEngine.Install.click()





