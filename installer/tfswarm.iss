; The Windows installer for TFSwarm (TASK-45), built by `.\build.ps1 -Installer`.
; It carries a CPU and a CUDA build of tfswarm.exe and installs the one chosen.

#ifndef AppVersion
  #define AppVersion "0.0.0"
#endif
#ifndef NumericVersion
  #define NumericVersion "0.0.0.0"
#endif
#ifndef CpuBin
  #define CpuBin "..\zig-out\cpu\bin"
#endif
#ifndef CudaBin
  #define CudaBin "..\zig-out\cuda\bin"
#endif
#define Repo "https://github.com/AIForGames2026/DistributedMoEEngine"

[Setup]
AppId={{B6F10BB4-FF11-406B-8526-5F50CEC6DF5A}
AppName=TFSwarm
AppVersion={#AppVersion}
AppVerName=TFSwarm {#AppVersion}
AppPublisher=TensorFold-Swarm
AppPublisherURL={#Repo}
AppSupportURL={#Repo}/issues
AppUpdatesURL={#Repo}/releases
VersionInfoVersion={#NumericVersion}
DefaultDirName={autopf}\TFSwarm
DefaultGroupName=TFSwarm
DisableProgramGroupPage=yes
; Installs for this user without admin rights, or for everyone when the user picks that.
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog commandline
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
MinVersion=10.0
OutputDir=..\dist
OutputBaseFilename=TFSwarm-Setup-{#AppVersion}
SetupIconFile=..\res\windows\tfswarm.ico
WizardSmallImageFile=..\res\windows\wizard-small.png
UninstallDisplayIcon={app}\tfswarm-operator.exe
UninstallDisplayName=TFSwarm
WizardStyle=modern
Compression=lzma2/max
SolidCompression=yes
SetupLogging=yes

[Messages]
FinishedLabel=Setup has installed TFSwarm. The first time it starts, it opens a setup page in your browser, where you choose whether this PC hosts the dashboard or joins one as a node. Its icon sits in the notification area.

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; Flags: unchecked
Name: "startup"; Description: "Start TFSwarm when signing in to Windows"; Flags: unchecked

[Files]
Source: "{#CpuBin}\tfswarm-operator.exe"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#CpuBin}\tfswarm.exe"; DestDir: "{app}"; Check: not UseCuda; Flags: ignoreversion
Source: "{#CudaBin}\tfswarm.exe"; DestDir: "{app}"; Check: UseCuda; Flags: ignoreversion
Source: "..\README.md"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\LICENSE"; DestDir: "{app}"; DestName: "LICENSE.txt"; Flags: ignoreversion
Source: "..\THIRD_PARTY.md"; DestDir: "{app}"; Flags: ignoreversion

[Icons]
Name: "{autoprograms}\TFSwarm"; Filename: "{app}\tfswarm-operator.exe"
Name: "{autodesktop}\TFSwarm"; Filename: "{app}\tfswarm-operator.exe"; Tasks: desktopicon
Name: "{autostartup}\TFSwarm"; Filename: "{app}\tfswarm-operator.exe"; Tasks: startup

[Run]
Filename: "{app}\tfswarm-operator.exe"; Description: "Start TFSwarm now"; Flags: nowait postinstall skipifsilent runasoriginaluser
Filename: "{app}\tfswarm-operator.exe"; Flags: nowait runasoriginaluser; Check: Relaunch

[Code]
var
  EnginePage: TInputOptionWizardPage;
  HasNvidia: Boolean;

function SwitchGiven(Name: String): Boolean;
var
  I: Integer;
begin
  Result := False;
  for I := 1 to ParamCount do
  begin
    if CompareText(ParamStr(I), Name) = 0 then
    begin
      Result := True;
    end;
  end;
end;

{ The tray launcher's update passes /RELAUNCH, so it starts again after a quiet install. }
function Relaunch: Boolean;
begin
  Result := SwitchGiven('/RELAUNCH');
end;

function UseCuda: Boolean;
begin
  Result := EnginePage.Values[0];
end;

procedure InitializeWizard;
var
  Choice: String;
begin
  HasNvidia := FileExists(ExpandConstant('{sys}\nvcuda.dll'));
  EnginePage := CreateInputOptionPage(wpSelectDir, 'Engine', 'Run models on an NVIDIA GPU, or on the CPU only?',
    'The CUDA engine needs an NVIDIA graphics card and its driver. The CPU engine runs on any PC. To change this later, run this installer again.',
    True, False);
  EnginePage.Add('CUDA: use the NVIDIA GPU, and load models on it by default');
  EnginePage.Add('CPU only');
  { /ENGINE=cuda or /ENGINE=cpu chooses from the command line; otherwise the last install's choice, or what this PC has. }
  Choice := Lowercase(ExpandConstant('{param:ENGINE|}'));
  if Choice = '' then
  begin
    Choice := GetPreviousData('Engine', '');
  end;
  if Choice = '' then
  begin
    if HasNvidia then
    begin
      Choice := 'cuda';
    end
    else
    begin
      Choice := 'cpu';
    end;
  end;
  EnginePage.Values[0] := Choice = 'cuda';
  EnginePage.Values[1] := Choice <> 'cuda';
end;

procedure RegisterPreviousData(PreviousDataKey: Integer);
begin
  if UseCuda then
  begin
    SetPreviousData(PreviousDataKey, 'Engine', 'cuda');
  end
  else
  begin
    SetPreviousData(PreviousDataKey, 'Engine', 'cpu');
  end;
end;

function UpdateReadyMemo(Space, NewLine, MemoUserInfoInfo, MemoDirInfo, MemoTypeInfo, MemoComponentsInfo, MemoGroupInfo, MemoTasksInfo: String): String;
var
  Engine: String;
begin
  if UseCuda then
  begin
    Engine := 'CUDA (NVIDIA GPU)';
  end
  else
  begin
    Engine := 'CPU only';
  end;
  Result := MemoDirInfo + NewLine + NewLine + 'Engine:' + NewLine + Space + Engine;
  if MemoTasksInfo <> '' then
  begin
    Result := Result + NewLine + NewLine + MemoTasksInfo;
  end;
end;

function NextButtonClick(CurPageID: Integer): Boolean;
begin
  Result := True;
  if (CurPageID = EnginePage.ID) and UseCuda and not HasNvidia then
  begin
    Result := MsgBox('No NVIDIA driver was found on this PC, so the CUDA engine will not start until one is installed. Install the CUDA engine anyway?', mbConfirmation, MB_YESNO or MB_DEFBUTTON2) = IDYES;
  end;
end;

{ Asks every running tray launcher to quit, which also ends its dashboard or node agent. }
procedure CloseLaunchers;
var
  Wnd: HWND;
  Tries: Integer;
begin
  Tries := 0;
  Wnd := FindWindowByClassName('TFSwarmOperatorTray');
  while (Wnd <> 0) and (Tries < 100) do
  begin
    PostMessage(Wnd, $0111, 7, 0);
    Sleep(100);
    Tries := Tries + 1;
    Wnd := FindWindowByClassName('TFSwarmOperatorTray');
    if Wnd = 0 then
    begin
      Sleep(500);
    end;
  end;
end;

function PrepareToInstall(var NeedsRestart: Boolean): String;
begin
  CloseLaunchers;
  Result := '';
end;

{ Settings, secrets and logs stay unless the user asks to delete them, or passes /DELETESETTINGS. }
procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
var
  Dir: String;
begin
  { After the user confirms, before any file goes. }
  if CurUninstallStep = usUninstall then
  begin
    CloseLaunchers;
  end;
  if CurUninstallStep <> usPostUninstall then
  begin
    exit;
  end;
  Dir := ExpandConstant('{localappdata}\TFSwarm');
  if not DirExists(Dir) then
  begin
    exit;
  end;
  if SwitchGiven('/DELETESETTINGS') then
  begin
    DelTree(Dir, True, True, True);
  end
  else if not UninstallSilent then
  begin
    if MsgBox('Also delete your TFSwarm settings, admin token, node secret and logs in ' + Dir + '?' + #13#10#13#10 + 'Keep them to reinstall later with the same setup.', mbConfirmation, MB_YESNO or MB_DEFBUTTON2) = IDYES then
    begin
      DelTree(Dir, True, True, True);
    end;
  end;
end;
