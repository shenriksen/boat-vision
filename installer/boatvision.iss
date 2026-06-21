; Inno Setup script for Boat Vision.
; Produces BoatVisionSetup.exe - a normal Windows installer (Next / Next / Finish).
; Per-user install (no admin needed); app data stays writable.

#define MyAppName "Boat Vision"
#define MyAppVersion "1.0"
#define MyAppExe "BoatVision.exe"

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
Source: "..\BoatVision.exe";        DestDir: "{app}"; Flags: ignoreversion
Source: "..\launcher.py";           DestDir: "{app}"; Flags: ignoreversion
Source: "..\requirements.txt";      DestDir: "{app}"; Flags: ignoreversion
Source: "..\README_WINDOWS.md";     DestDir: "{app}"; Flags: ignoreversion
Source: "..\yolo26n.pt";            DestDir: "{app}"; Flags: ignoreversion
Source: "..\boat_vision\*";         DestDir: "{app}\boat_vision"; Flags: recursesubdirs ignoreversion
Source: "..\configs\*";             DestDir: "{app}\configs";     Flags: recursesubdirs ignoreversion
Source: "..\models\*";              DestDir: "{app}\models";      Flags: recursesubdirs createallsubdirs ignoreversion
Source: "..\docs\*";                DestDir: "{app}\docs";        Flags: recursesubdirs ignoreversion
Source: "postinstall.ps1";          DestDir: "{app}\installer";   Flags: ignoreversion

[Icons]
Name: "{group}\{#MyAppName}";        Filename: "{app}\{#MyAppExe}"; IconFilename: "{app}\boat_vision\static\app_icon.ico"
Name: "{autodesktop}\{#MyAppName}";  Filename: "{app}\{#MyAppExe}"; IconFilename: "{app}\boat_vision\static\app_icon.ico"

[Run]
Filename: "powershell.exe"; \
  Parameters: "-NoProfile -ExecutionPolicy Bypass -File ""{app}\installer\postinstall.ps1"""; \
  StatusMsg: "Installing AI components (downloads ~2.5 GB - please wait, this can take 10+ minutes)..."; \
  Flags: waituntilterminated
Filename: "{app}\{#MyAppExe}"; Description: "Start Boat Vision now"; Flags: postinstall nowait skipifsilent
