function [ xmlInfo ] = parseoxy3header(xmlData, myString)
% Parses the meta information from an .oxy3-file.
persistent prevXmlInfo;
persistent prevXmlData;

if nargin <2
  % read xmlData
  myString = {'Version', 'License', 'Date', 'Device', 'Laser', 'APD' 'nTx', 'nRx', 'nADC', ...
    'nbSamples','OptodeTemplateID', 'SampleTime', 'Wavelength', 'dark', 'DPF', 'Position', ...
    'abs_K', 'abs_H', 'Gradient', 'Event', 'Name','Description','laserpower',...
    'gain','GainEvent','PwrEvent','abs_H2O', 'pos1_width', 'pos2_width','use_H2O', ...
  'fileType', 'NirsOffsets', 'AdChName', 'dataFormat', 'SampleRate', 'CreateDate'};
else
  if ~ismember(myString, 'Device')
    myString{2:end+1} = myString;
    myString{1} = 'Device';
  end
end
% for better debugging:
% %           1           2           3       4       5           6   7      8     9
% myString = {'Version', 'License', 'Date', 'Device', 'Laser', 'APD' 'nTx', 'nRx', 'nADC'};
% myString = [myString ...
%     {'nbSamples','OptodeTemplateID', 'SampleTime', 'Wavelength', 'dark', 'DPF', 'Position'}];
% %           10           11         12              13          14       15      16
% myString = [myString ...
%     {'abs_K', 'abs_H', 'Gradient', 'Event', 'Name',' Description','laserpower'}];
% %    17       18       19          20       21       22             23
% myString = [myString ...
%     {'gain','GainEvent','PwrEvent','abs_H2O', 'pos1_width', 'pos2_width','use_H2O'}];
% %    24      25         26          27        28            29              30

persistent prevMyString;

if isequal(prevXmlData, xmlData) && isequal(prevMyString, myString) && ~isempty(prevXmlInfo)
  xmlInfo = prevXmlInfo;
  return;
else
  prevXmlData = xmlData;
  prevMyString = myString;
end

xmlInfo = struct('Format', {'OXY3'});
xmlInfo.index=[];
% j = strfind(myString, 'Position'); % debugging code

for i = 1:length(xmlData)
  for j = 1:length(myString)
    
    e = min(i+length(myString{j}+1), length(xmlData));
    if strcmp(xmlData(i:e), ['<' myString{j}])
      % Check whether the header parser won't take metadata (CyclicEvent)as Events or
      % miss Events in Oxy5
      if strcmp(myString{j},'Event')
        if strcmp(xmlData(e+1:e+5),' Time')
            e = min(i+50, length(xmlData));
            k1 = find(xmlData(i+1:e) == '"', 1, 'first');
            k2 = find(xmlData(i+k1+1:e) == '"', 1, 'first');
            info = xmlData(i+k1+1:i+k1+k2-1);
        elseif strcmp(xmlData(e+1:e+3),' ID')
            e = min(i+50, length(xmlData));
            k1 = find(xmlData(i+1:e) == '"', 3, 'first');
            k1 = k1(3);
            k2 = find(xmlData(i+k1+1:e) == '"', 1, 'first');
            info = xmlData(i+k1+1:i+k1+k2-1);
        elseif strcmp(xmlData(e+1:e+11),'CyclicTrace')
            break;
        else
        end
      else 
        e = min(i+50, length(xmlData));
        k1 = find(xmlData(i+1:e) == '>', 1, 'first');
        k2 = find(xmlData(i+1:e) == '<', 1, 'first');
        if ~isempty(xmlData(i+k1+1:i+k2-1)) && ~isempty(strtrim(xmlData(i+k1+1:i+k2-1)))
          info =  xmlData(i+k1+1:i+k2-1);
        else
          k1 = find(xmlData(i+1:e) == '"', 1, 'first');
          k2 = find(xmlData(i+k1+1:e) == '"', 1, 'first');
          info = xmlData(i+k1+1:i+k1+k2-1);
        end
      end
      if isempty(info)
        info = '';
      elseif ~isempty(str2double(info)) && strcmp(num2str(str2double(info)),info)
        info = str2double(info);
      end
      
      if strcmp(myString{j},'OptodeTemplateID')
        % check whether this is in a AbsOffset tag
        if strfind(xmlData(i-50:e), '<AbsOffset')
          continue; % yep, it is, better skip
        end
      end
      
      if numel(strfind(xmlData(1:e), '<DeviceSelection')) > numel(strfind(xmlData(1:e), '</DeviceSelection'))
        % if this tag is within the DeviceSelection tag, skip it
        continue;
      end
      if numel(strfind(xmlData(1:e), '<LaserMapping')) > numel(strfind(xmlData(1:e), '</LaserMapping'))
        % if this tag is within the LaserMapping tag, skip it
        continue;
      end
        
      if strcmp(myString{j},'Device') && ~contains(xmlData(i:e), 'DeviceFWVersion')        
        k1 = strfind(xmlData(i:e+50), 'index');
        e = min(i+k1+10, e);
        k2 = strfind(xmlData(i+k1:e+50), '"');
        if numel(k2) > 2
            k2 = k2(1:2);
        end
        
        if ~isempty(k1) && numel(k2)==2
          xmlInfo.index = [xmlInfo.index, str2double(xmlData(i+k1+k2(1):i+k1+k2(2)-2))];
        else
          % if 'index' is not present, then we're screwed...
          % assume an unknown index -1 (which means, all
          % devices are interconnected, see masterDevices
          % variable
          xmlInfo.index = [xmlInfo.index, -1];
        end
      end
      
      if strcmp(myString{j},'Laser')
          stack_start = strfind(xmlData(i:e), 'Stack="');
          if ~isempty(stack_start)
              k1 = i+stack_start+6;
              k2 = find(xmlData(k1:e) == '"', 1, 'first');
              info = str2double(xmlData(k1:k1+k2-2));
          end         
      end
      
      if strcmp(myString{j},'Event')
        descr = '';
        evend = strfind(xmlData(i:end), '/Event');
        % check whether Description is present, then use this:
        descrbeg = strfind(xmlData(i:i+evend(1)), '<Description>');
        if ~isempty(descrbeg)
          descrend = strfind(xmlData(i:i+evend(1)), '</Description>');
          descr = xmlData(i+descrbeg+12:i+descrend-2);
          if ~isempty(descr)
            if ~isfield(xmlInfo , 'Description')
              xmlInfo.Description = {descr};
            else
              xmlInfo.Description{end+1} = descr;
            end
          end
        end
        
        name = '';
        namebeg = strfind(xmlData(i:i+evend(1)), '<Name>');
        if ~isempty(namebeg)
          nameend = strfind(xmlData(i:i+evend(1)), '</Name>');
          name = xmlData(i+namebeg+5:i+nameend-2);
          if ~isfield(xmlInfo, 'Name')
            xmlInfo.Name = {name};
          else
            xmlInfo.Name{end+1} = name;
          end
        elseif isempty(descr) % if name and descr not found, ignore (probably offline ignored event)
          if ~isfield(xmlInfo , 'Description')
            xmlInfo.Description = {''};
          else
            xmlInfo.Description{end+1} = '';
          end
        end
      end
      
      if strcmp(myString{j},'NirsOffsets')
		    offset = '';
        noend = strfind(xmlData(i:end), '/NirsOffsets');
        % check whether Description is present, then use this:
        valuebeg = strfind(xmlData(i:i+noend(1)), '<Value>');
        valueend = strfind(xmlData(i:i+noend(1)), '</Value>');
        if ~isempty(valuebeg)
          for v=1:numel(valuebeg)
            offset = str2double(xmlData(i+valuebeg(v)+6:i+valueend(v)-2));
            if ~isempty(offset)
              if ~isfield(xmlInfo , 'Offsets')
                xmlInfo.Offsets = [offset];
              else
                xmlInfo.Offsets(end+1) = offset;
              end
            end
          end
        end
      end
      
      if strcmp(myString{j},'AdChName')
        adchend = strfind(xmlData(i:end), '/AdChName');
        
        % extract adc name
        namebeg = strfind(xmlData(i:i+adchend(1)), '<Name>');
        nameend = strfind(xmlData(i:i+adchend(1)), '</Name>');
        if ~isfield(xmlInfo , 'ADCNames')
          xmlInfo.ADCNames{1} = xmlData(i+namebeg+5:i+nameend-2);
        else
          xmlInfo.ADCNames{end+1} = xmlData(i+namebeg+5:i+nameend-2);
        end
        continue; % skip the rest of this loop
      end
      
      if strcmp(myString{j},'Name') % ignore if it is an AdChName, not an Event Name
        %         if ~isempty(strfind(xmlData(i-25:i), 'AdChName')) % we got a wrong Name here
        break;
        %         end
      end
      
      if strcmp(myString{j},'Description')
        %         if isempty(info) && isempty(strfind(xmlData(i:i+45), 'Event'))
        % skip this one, as it is obviously some error
        break;
        %         end
      end
      
      %%%
      if strcmp(myString{j},'GainEvent')||strcmp(myString{j},'PwrEvent')
        k1 = find(xmlData(i+1:i+50) == '"', 6, 'first');
        k2 = find(xmlData(i+k1+1:i+50) == '"', 6, 'first');
        info= xmlData(i+k1(1):i+k1(1)+k2(end));
      end
      %%%
      
      
      if strcmp(myString{j},'SampleTime') 
        if ischar(info) % make sure this is a number;
          info = str2double(info);
        end
        
        xmlInfo.SampleTime = info; % there can be only one.
        continue;
      end
      
      if strcmp(myString{j},'SampleRate') % make sure this is a number;
        xmlInfo.SampleTime = 1/info;
        xmlInfo.SampleRate = info;
        continue;
      end
      
      if isfield(xmlInfo,myString{j})
        if ~iscell(xmlInfo.(myString{j}))
          info2 = xmlInfo.(myString{j});
          xmlInfo = rmfield(xmlInfo, myString{j});
          xmlInfo.(myString{j}){1} = info2;
        end
        k = length(xmlInfo.(myString{j}))+1;
        xmlInfo.(myString{j}){k} = info;
      else
        xmlInfo.(myString{j}) = info;
        clear k;
      end
      
      clear info
    end
    
  end
end

%if isfield(xmlInfo, 'Event') && ~iscell(xmlInfo.Event)
%  xmlInfo.Event = {xmlInfo.Event};
%end

%if isfield(xmlInfo, 'Description') && ~iscell(xmlInfo.Description)
%  xmlInfo.Event = {xmlInfo.Event};
%end

%if isfield(xmlInfo, 'Name') && ~iscell(xmlInfo.Name)
%  xmlInfo.Event = {xmlInfo.Event};
%end

%bUseDescr = isfield(xmlInfo, 'Description') && ~isempty(xmlInfo.Description) && numel(xmlInfo.Description) == numel(xmlInfo.Event);
%bUseName = isfield(xmlInfo, 'Name') && isfield(xmlInfo, 'Event') && ~bUseDescr;


% if bUseDescr
%   xmlInfo.Name = xmlInfo.Description;
% elseif isfield(xmlInfo, 'Description')
%   xmlInfo = rmfield(xmlInfo, 'Description');
% end

%if bUseName

if isfield(xmlInfo, 'Name')    
  if ~iscell(xmlInfo.Name)
    xmlInfo.Name = {xmlInfo.Name};
  end
  
  if numel(xmlInfo.Name) > numel(xmlInfo.Event)
    xmlInfo.Name=xmlInfo.Name(end-length(xmlInfo.Event)+1:end);
    if iscell(xmlInfo.Name)
      xmlInfo.Name = cellfun(@num2str, xmlInfo.Name, 'UniformOutput', false);
    end
  end
end

% Below code doesn't do anything, xmlInfo.abs.H2O never exists
% if isfield(xmlInfo, 'use_H2O')
%     if xmlInfo.use_H2O==0;
%         clear xmlInfo.abs.H2O
%     end
% end

% Stacks are seen as multiple systems
if isfield(xmlInfo, 'Version') && xmlInfo.Version (1)=='2'
  xmlInfo.index = zeros(length(xmlInfo.Device),1);
end

if isfield(xmlInfo, 'Device')
  if ~iscell(xmlInfo.Device)
    masterDevices = xmlInfo.Device(1);
    if numel(xmlInfo.Device)>1
      %fprintf('whoopa, what happen here, mate?\n');
      %fprintf('can we have multiple devices in an array instead of a cell?\n');
      error('[Artinis] sorry, never had this before, please contact jorn@artinis.com and send him a snippet of your data. Thanks!');
    end
  else
    indices = unique(xmlInfo.index,'stable');
    masterDevices = [];
    j = 1; % we need this index to handle LSL streams
    for i=1:numel(indices)
      idx = find(xmlInfo.index==indices(i));
      if all(([xmlInfo.Device{idx}] == 1234)>0)
        for k=1:numel(idx)
          masterDevices{j} = sprintf('%i', xmlInfo.Device{idx(k)});
          j = j+1;
        end
        continue; % this is an LSL stream, we do not want to merge LSL streams
      end
      masterDevices{j} = sprintf('%i_', xmlInfo.Device{idx});
      masterDevices{j} = masterDevices{j}(1:end-1);
      j = j+1;
    end
  end
  
  
  % above code should be reproducing what the below commented code should do
  % masterDevices=xmlInfo.Device(1);
  % for i=2:length(xmlInfo.index)
  %     if xmlInfo.index(i-1)~=xmlInfo.index(i)
  %         masterDevices=[masterDevices xmlInfo.Device(i)];
  %     else
  %         masterDevices{i-1}= strcat(num2str(xmlInfo.Device{i-1}),'_', num2str(xmlInfo.Device{i}));
  %     end
  % end
  
  xmlInfo.Device=masterDevices;
else
  xmlInfo.Device=[];
end

if isfield('APD', xmlInfo) && isscalar(xmlInfo.APD)
  xmlInfo.APD = {xmlInfo.APD};
end
  
if isfield(xmlInfo, 'Laser') && isscalar(xmlInfo.Laser)
  xmlInfo.Laser = {xmlInfo.Laser};
end

if isfield(xmlInfo, 'OptodeTemplateID')
  % TSI files have two IDs in the oxy3 file. First IDs are the TSI cali ID
  if length(xmlInfo.OptodeTemplateID)>length(xmlInfo.Device)      
    if length(xmlInfo.OptodeTemplateID) == 2 * length(xmlInfo.Device)
        % probably a bug with double optode IDs in the xml header, so
        % remove one half
        xmlInfo.OptodeTemplateID = xmlInfo.OptodeTemplateID(1:length(xmlInfo.Device));
    elseif any([xmlInfo.OptodeTemplateID{:}] == 999)
      % remove the PS/ADBox/EEG template
      xmlInfo.OptodeTemplateID([xmlInfo.OptodeTemplateID{:}] == 999) = [];
    else
      % TSI files have two IDs in the oxy3 file. First IDs are the TSI cali ID
      xmlInfo.OptodeTemplateID=cell2mat(xmlInfo.OptodeTemplateID(end-length(xmlInfo.Device)+1:end));
    end
  else
    % it seems to me that 2 stacks have three devices somehow...
    % [sys1-sys2 sys3 sys3-sys4]
    % I do not understand why, yet
    xmlInfo.OptodeTemplateID=xmlInfo.OptodeTemplateID(end-length(xmlInfo.Device)+1:end);
  end
else
  xmlInfo.OptodeTemplateID = -1;
end

% fix for PM/PL offline data, which does not have an optodeparametertable
if ~isfield(xmlInfo, 'Position')
  warning('No nominal distances defined in oxy-file. Assuming a nominal distance of 35mm.');
  xmlInfo.Position = 35;
end


if ~isfield(xmlInfo, 'Gradient')
  warning('No nominal distance gradient defined in oxy-file. Assuming a nominal distance gradient of 5mm.');
  xmlInfo.Gradient = 5;
end

if ~isfield(xmlInfo, 'abs_H')
  if ~iscell(xmlInfo.OptodeTemplateID) & xmlInfo.OptodeTemplateID == 28 | xmlInfo.OptodeTemplateID == 29 | xmlInfo.OptodeTemplateID == 147
    warning('No abs_H parameter defined in oxy-file. Assuming an abs_H for muscle measurements.');
    xmlInfo.abs_H = 0.00046;
  else
    warning('No abs_H parameter defined in oxy-file. Assuming an abs_H for brain measurements.');
    xmlInfo.abs_H = 0.00045;
  end
end

if ~isfield(xmlInfo, 'abs_K')
  if ~iscell(xmlInfo.OptodeTemplateID) | xmlInfo.OptodeTemplateID == 28 | xmlInfo.OptodeTemplateID == 29 | xmlInfo.OptodeTemplateID == 147
    warning('No abs_K parameter defined in oxy-file. Assuming an abs_K for muscle measurements.');
    xmlInfo.abs_K= 1.1;
  else
    warning('No abs_K parameter defined in oxy-file. Assuming an abs_K for brain measurements.');
    xmlInfo.abs_K= 1.45;
  end
end

if ~isfield(xmlInfo, 'DPF')
  if ~iscell(xmlInfo.OptodeTemplateID) | xmlInfo.OptodeTemplateID == 28 | xmlInfo.OptodeTemplateID == 29 | xmlInfo.OptodeTemplateID == 147
    warning('No DPF parameter defined in oxy-file. Assuming a DPF for muscle measurements.');
    xmlInfo.DPF = 4;
  else
    warning('No DPF parameter defined in oxy-file. Assuming a DPF for brain measurements.');
    xmlInfo.DPF = 6;
  end
end

prevXmlInfo = xmlInfo;
return;

