; Inno Setup script for Boat Vision (GPU edition).
; Produces BoatVisionSetup.exe - a normal Windows installer (Next / Next / Finish).
; Per-user install (no admin). At install time it sets up Python + the CUDA
; (GPU) PyTorch build, so it runs full speed on an NVIDIA GPU (e.g. RTX 4060).
; The app opens in its own native window (pywebview) - not a browser.

#ifndef MyAppVersion
  #define MyAppVersion "1.1.0"
#endif
#define MyAppName "Boat Vision"

[Setup]
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher=Nordic USV
DefaultDirName={localappdata}\Programs\BoatVision
DefaultGroupName={#MyAppName}
DisableProgramGroupPage=yes
OutputDir=Output
OutputBaseFilename=BoatVisionSetup
SetupIconFile=..\boat_vision\static\app_icon.ico
Compression=lzma2
SolidCompression=yes
PrivilegesRequired=lowest
WizardStyle=modern

[Files]
Source: "..\app_native.py";       DestDir: "{app}"; Flags: ignoreversion
Source: "..\launcher.py";         DestDir: "{app}"; Flags: ignoreversion
Source: "..\requirements.txt";    DestDir: "{app}"; Flags: ignoreversion
Source: "..\README_WINDOWS.md";   DestDir: "{app}"; Flags: ignoreversion
Source: "..\VERSION";             DestDir: "{app}"; Flags: ignoreversion
Source: "..\yolo26n.pt";          DestDir: "{app}"; Flags: ignoreversion
Source: "..\yolo26s.pt";          DestDir: "{app}"; Flags: ignoreversion
Source: "..\boat_vision\*";       DestDir: "{app}\boat_vision"; Flags: recursesubdirs ignoreversion
Source: "..\configs\*";           DestDir: "{app}\configs";     Flags: recursesubdirs ignoreversion
Source: "..\models\*";            DestDir: "{app}\models";      Flags: recursesubdirs createallsubdirs ignoreversion
Source: "..\docs\*";              DestDir: "{app}\docs";        Flags: recursesubdirs ignoreversion
Source: "postinstall.ps1";        DestDir: "{app}\installer";   Flags: ignoreversion
Source: "python-3.12.7-amd64.exe"; DestDir: "{app}\installer";  Flags: ignoreversion

[Icons]
Name: "{group}\{#MyAppName}";       Filename: "{app}\.venv\Scripts\pythonw.exe"; Parameters: """{app}\app_native.py"""; WorkingDir: "{app}"; IconFilename: "{app}\boat_vision\static\app_icon.ico"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\.venv\Scripts\pythonw.exe"; Parameters: """{app}\app_native.py"""; WorkingDir: "{app}"; IconFilename: "{app}\boat_vision\static\app_icon.ico"

[Run]
Filename: "powershell.exe"; \
  Parameters: "-NoProfile -ExecutionPolicy Bypass -File ""{app}\installer\postinstall.ps1"""; \
  StatusMsg: "Installing AI components (downloads PyTorch GPU build, ~2.5 GB - this can take 10+ minutes)..."; \
  Flags: waituntilterminated
Filename: "{app}\.venv\Scripts\pythonw.exe"; Parameters: """{app}\app_native.py"""; WorkingDir: "{app}"; \
  Description: "Start Boat Vision now"; Flags: postinstall nowait skipifsilent
