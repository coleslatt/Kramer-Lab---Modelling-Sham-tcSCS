%% AUC_MEP_withOverlay_LamKramer
% by Jason Mansour Anasori
% Jan 21, 2026
%% daq info and shit like that
Fs = 2000;                 % Hz
gain = 909;                % Delsys gain
to_mV = 1000;              % V -> mV

stimDerivThr = 1;          % derivative threshold for stim detection
delay_s = 0.060;           % 60 ms Trigno analog delay
delay_frames = round(delay_s * Fs);

bgWin = [-0.11, -0.01];    % background window (-110 ms to -10 ms)

% MEP windows (s) --> change them after looking at your overlay plot
win_RF = [0.03, 0.09];     % RRF/LRF (15 ms to 200 ms)
win_TA = [0.030, 0.09];     % RTA/LTA (30 ms to 200 ms)

% Overlay time window (s)
tLim  = [-0.10, 0.30];
%% select ALL trial CSVs
[fileList, path] = uigetfile({'*.csv','CSV files (*.csv)'}, ...
    'Select ALL trial CSVs (e.g., 140 files)', 'MultiSelect', 'on');

if isequal(fileList,0)
    error('No files selected.');
end
if ischar(fileList) || isstring(fileList)
    fileList = cellstr(fileList);
end

fileList = sort(fileList);
nFiles = numel(fileList);

fprintf("\nSelected %d files.\n", nFiles);
if nFiles ~= 140
    warning("Expected 140 trials, but you selected %d. Labels will still be applied in order.", nFiles);
end
%% trial labeling (your pilot order)
nCfg = 3;                 % number of configs
trialsPerCfg = 40;        % number of trials in each config
baselineTrials = 20;      % last 20 are baseline

mainTrials = nCfg * trialsPerCfg;             % 120
baselineStart = mainTrials + 1;               % 121
baselineEnd   = mainTrials + baselineTrials;  % 140

TrialNum = (1:nFiles)';

Config = strings(nFiles,1);
MSO   = nan(nFiles,1);      % 63 or 76 (kept if you want later)
Stim  = false(nFiles,1);    % WITH stim = true, NO stim = false
Block = strings(nFiles,1);  % helpful label

for i = 1:nFiles
    if i <= mainTrials
        cfg = ceil(i/trialsPerCfg);               % which config
        within = mod(i-1,trialsPerCfg) + 1;       % 1..40
        Config(i) = "Config" + cfg;

        if within <= 10
            MSO(i) = 63; Stim(i) = false; Block(i) = "63_noStim";
        elseif within <= 20
            MSO(i) = 76; Stim(i) = false; Block(i) = "76_noStim";
        elseif within <= 30
            MSO(i) = 76; Stim(i) = true;  Block(i) = "76_stim";
        else
            MSO(i) = 63; Stim(i) = true;  Block(i) = "63_stim";
        end

    elseif i >= baselineStart && i <= baselineEnd
        Config(i) = "Baseline";
        within = i - mainTrials;                  % 1..20

        if within <= 10
            MSO(i) = 63; Stim(i) = false; Block(i) = "63_noStim";
        else
            MSO(i) = 76; Stim(i) = false; Block(i) = "76_noStim";
        end

    else
        Config(i) = "Extra";
        Block(i)  = "Unlabeled";
        Stim(i)   = false;
        MSO(i)    = NaN;
    end
end
%% preallocate outputs
emgNames = ["RTA","LTA","RRF","LRF"];
AUC_net  = nan(nFiles,4);
StimDetected = false(nFiles,1);

% overlay storage (rectified baseline-corrected)
tGrid = (tLim(1):1/Fs:tLim(2))';
nT = numel(tGrid);
RECT_NET   = nan(nT,4,nFiles);
validTrace = false(nFiles,1);

dt = 1/Fs;
%% initialize CSV import template
firstFile = fullfile(path, fileList{1});
optsTemplate = detectImportOptions(firstFile, 'NumHeaderLines', 11);
optsTemplate.VariableNamesLine = 12;
optsTemplate.DataLines = [13 Inf];
%% main loop
for ii = 1:nFiles
    fname = fullfile(path, fileList{ii});
    fprintf("(%3d/%3d) %s\n", ii, nFiles, fileList{ii});

    % read table
    T = readtable(fname, optsTemplate);
    raw = table2array(T);
    ch  = lower(string(T.Properties.VariableNames));

    % map channels
    idx_tms = find(ch == "tms", 1);
    idx_rta = find(ch == "rta", 1);
    idx_lta = find(ch == "lta", 1);
    idx_rrf = find(ch == "rrf", 1);
    idx_lrf = find(ch == "lrf", 1);

    if any([isempty(idx_tms), isempty(idx_rta), isempty(idx_lta), isempty(idx_rrf), isempty(idx_lrf)])
        warning("Missing channel(s) in %s. Skipping.", fileList{ii});
        continue;
    end

    emgCh = [idx_rta, idx_lta, idx_rrf, idx_lrf];

    % detect stim
    N      = size(raw,1);
    tmsSig = raw(:, idx_tms);

    stimIdx = find(abs(diff(tmsSig)) > stimDerivThr, 1, 'first');
    if isempty(stimIdx)
        StimDetected(ii) = false;
        if Stim(ii)
            warning("No stim detected in a trial labeled WITH stim: %s. Skipping.", fileList{ii});
        end
        continue;
    end
    StimDetected(ii) = true;

    % time vector centered on stim
    t = ((1:N) - stimIdx) / Fs;

    % shift EMG earlier to compensate Trigno delay
    if delay_frames >= N
        warning("Delay frames >= trial length in %s. Skipping.", fileList{ii});
        continue;
    end
    emgV = raw(:, emgCh);
    emgV_shifted = nan(size(emgV));
    emgV_shifted(1:N-delay_frames,:) = emgV((delay_frames+1):N,:);

    % baseline window indices
    bgIdx = find(t >= bgWin(1) & t <= bgWin(2));
    if numel(bgIdx) < 10
        warning("Too few baseline samples in %s. Skipping.", fileList{ii});
        continue;
    end

    % baseline offset + gain correction (mV)
    base   = mean(emgV_shifted(bgIdx,:), 1);
    emg_mV = to_mV * (emgV_shifted - base) / gain;

    % rectify + rectified baseline floor
    rect_mV    = abs(emg_mV);
    bgRectMean = mean(rect_mV(bgIdx,:), 1);

    % AUC_net
    for k = 1:4
        if k <= 2
            mepWin = win_TA;   % TA channels
        else
            mepWin = win_RF;   % RF channels
        end

        mepIdx = find(t >= mepWin(1) & t <= mepWin(2));
        if numel(mepIdx) < 10
            warning("MEP window too short in %s (%s).", fileList{ii}, emgNames(k));
            continue;
        end

        auc_total = trapz(t(mepIdx), rect_mV(mepIdx,k));   % mV*s
        mepDur    = (numel(mepIdx)-1) * dt;                % s
        auc_base  = bgRectMean(k) * mepDur;                % mV*s

        AUC_net(ii,k) = auc_total - auc_base;              % baseline-corrected rectified AUC
    end

    % store overlay trace: rectified baseline-corrected
    rectNet = rect_mV - bgRectMean;     % subtract baseline floor
    rectNet(rectNet < 0) = 0;           % clip negatives

    for k = 1:4
        RECT_NET(:,k,ii) = interp1(t, rectNet(:,k), tGrid, 'linear', nan);
    end
    validTrace(ii) = true;
end

fprintf("\nStored overlay traces for %d/%d trials.\n", sum(validTrace), nFiles);
%% save trial-level results
Results = table();
Results.TrialNum = TrialNum;
Results.FileName = string(fileList(:));
Results.Config   = Config;
Results.Block    = Block;

% Uncomment if you want these in output (i dont think you need them)
% Results.MSO        = MSO;
% Results.StimLabel  = Stim;
% Results.StimDetected = StimDetected;

Results.AUC_RTA = AUC_net(:,1);
Results.AUC_LTA = AUC_net(:,2);
Results.AUC_RRF = AUC_net(:,3);
Results.AUC_LRF = AUC_net(:,4);

outCSV = fullfile(path, "AUC_results_allTrials.csv");
writetable(Results, outCSV);
fprintf("\nSaved trial-level results:\n%s\n", outCSV);
%% save 10-trial summary (config x block)
[G, cfgU, blkU] = findgroups(Results.Config, Results.Block);

MeanAUC_RTA = splitapply(@(x) mean(x,'omitnan'), Results.AUC_RTA, G);
MeanAUC_LTA = splitapply(@(x) mean(x,'omitnan'), Results.AUC_LTA, G);
MeanAUC_RRF = splitapply(@(x) mean(x,'omitnan'), Results.AUC_RRF, G);
MeanAUC_LRF = splitapply(@(x) mean(x,'omitnan'), Results.AUC_LRF, G);

N_RTA = splitapply(@(x) sum(~isnan(x)), Results.AUC_RTA, G);
N_LTA = splitapply(@(x) sum(~isnan(x)), Results.AUC_LTA, G);
N_RRF = splitapply(@(x) sum(~isnan(x)), Results.AUC_RRF, G);
N_LRF = splitapply(@(x) sum(~isnan(x)), Results.AUC_LRF, G);

Summary10 = table(cfgU, blkU, ...
    N_RTA, N_LTA, N_RRF, N_LRF, ...
    MeanAUC_RTA, MeanAUC_LTA, MeanAUC_RRF, MeanAUC_LRF, ...
    'VariableNames', {'Config','Block', ...
                      'N_RTA','N_LTA','N_RRF','N_LRF', ...
                      'MeanAUC_RTA','MeanAUC_LTA','MeanAUC_RRF','MeanAUC_LRF'});

outCSV2 = fullfile(path, "AUC_summary_10trials.csv");
writetable(Summary10, outCSV2);
fprintf("\nSaved 10-trial summary:\n%s\n", outCSV2);
%% plot overlay (rectified baseline-corrected)
figure('Units','normalized','OuterPosition',[0 0 1 1]);
tiledlayout(2,2,'Padding','compact','TileSpacing','compact');

for k = 1:4
    nexttile; hold on;

    for ii = find(validTrace)'
        plot(tGrid, RECT_NET(:,k,ii), 'Color', [0.75 0.75 0.75]);
    end

    m = mean(RECT_NET(:,k,validTrace), 3, 'omitnan');
    plot(tGrid, m, 'k', 'LineWidth', 2);

    xline(0,'k--','LineWidth',1.2);

    if k<=2
        w = win_TA;
    else
        w = win_RF;
    end
    xline(w(1),'b--','LineWidth',1.2);
    xline(w(2),'b--','LineWidth',1.2);

    title(sprintf('%s | rectified baseline-corrected overlay (n=%d)', emgNames(k), sum(validTrace)));
    xlabel('Time (s)');
    ylabel('|EMG|_{net} (mV)');
    xlim(tLim);
    grid on;
end
