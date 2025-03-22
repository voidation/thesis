function [pos] = oxysoftMNIImport(filename, target, savename, orderFile)
% This function takes an Oxysoft Excel-export of MNI-coordinates and
% stores them in appropriate format and order to be used in NIRS-SPM
% , AtlasViewer or NIRSTORM
% Use as
%   [pos] = oxysoftMNIImport(filename, [target], [savename], [orderFile])
% where 
%   filename  - char, an Excel-file obtained from Oxysoft's Calculate dialog
%   target    - string, the target toolbox. Can be 'nirs-spm', 'atlasviewer'
%  or 'nirstorm'
%   savename  - char, the destination, where the output should be stored.
%   orderFile - char, (for 'nirs-spm' only) the txt-file specifying the order of the 
%         optodes or channels retrieved from Artinis' oxysoft2matlab function
%
% All input arguments can be empty or left out, in which case dialog-boxes
% will ask for appropriate specification.
%
% Version 1.5, copyright (c) by Artinis Medical Systems http://www.artinis.com
% Author Jrn M. Horschig, jorn@artinis.com
% Author Saskia Kruitwagen
% Author Mathijs Bronkhorst, mathijs@artinis.com
% Author Kristoffer Dahlslatt kristoffer@artinis.com
% See also OXYSOFT2MATLAB

%%
% Version history
%
% v 1.0     org - initial version
% 08/26/16
%
% v 1.1     enh - atlasviewer supported (by Saskia Kruitwagen)
% 01/15/18 
%
% v 1.2     fix - nirs-spm support for new 3D export (fiducials)
% 06/22/18  fix - removing file extension on call to xlsread as this made 
%             trouble on some platforms
%           fix - correctly forcing atlasviewer export to be named 'digpts'
% v 1.3
% 12/10/20  enh - Added nirstorm support (Kristoffer Dahlslatt)
%
% v 1.4
% 04/18/23  fix - Atlasviewer conversion can handle unordered input
%
% v 1.5 
% 03/11/24  fix - Atlasviewer conversion compatible with >=Matlab 2022a
%           fix - Atlasviewer conversion can handle multi-device measurements

%% if no input argument is provided, show a dialog
if nargin < 1 || isempty(filename)
  [filename,pathname] = uigetfile({'*.xls; *.xlsx', 'Microsoft Excel'}, 'Choose an Oxysoft MNI-Export Excel-file');
  filename = fullfile(pathname, filename);
end

% check whether the file is actually an Excel file
[p, n, e] = fileparts(filename);
if ~strcmp(e, '.xls') && ~strcmp(e, '.xlsx')
  error('You have to specify an Oxysoft MNI-Export Excel-file to use this function.');
end

% check whether target toolbox is supported
supported_targets = {'nirs-spm', 'atlasviewer', 'nirstorm'};
if nargin < 2 || isempty(target )
  target_idx = menu('Choose a target toolbox',supported_targets);
  target     = supported_targets{target_idx};
  drawnow;
  fprintf('[Artinis] Selected target toolbox %s.\n', target);
end

if ~ismember(target, supported_targets)
  tgts = sprintf('''%s'', ', supported_targets{1:end-1});
  tgts = sprintf('%s and ''%s''', tgts(1:end-2), supported_targets{end});
  error('[Artinis] target toolbox cannot be identified. please choose among %s.', tgts);
end

switch(target)
  
  case {'nirs-spm'}

    %% retrieve the order of the optodes or channels
    if nargin < 4 || isempty(orderFile)
      [orderFile,pathname] = uigetfile({'*optode_order.txt; *channel_order.txt', 'Text Document'}, 'Choose an Optode- or Channel-Order text file obtained from oxysoft2matlab');
      orderFile = fullfile(pathname, orderFile);
    end

    isChanOrder = ~isempty(strfind(orderFile, 'channel_order.txt'));
    isOptoOrder = ~isempty(strfind(orderFile, 'optode_order.txt'));

    if isempty(orderFile) || (~isChanOrder && ~isOptoOrder)
      error('No optode- or channel-order file specified. Please use the oxysoft2matlab script and choose ''nirs-spm'' as output toolbox to create such files.\n');
    end

    fid = fopen(orderFile);
    order = textscan(fid, '%d %d');
    fclose(fid);

    %% prepare local storage of the output
    if nargin < 3
      [savename,pathname] = uiputfile({'*.txt', 'Text Document'}, 'Save Oxysoft MNI-Export');
      savename = fullfile(pathname, savename);
    end

    %% read the file
    data = xlsread(fullfile(p, n));
    nanstart = find(isnan(data(:, 1)));

    %% extract the positions, reorder and save
    if isChanOrder
      pos = data(nanstart(4)+1:end, :);
      pos(order{1}, :) = pos(order{2}, :);
      if ~isempty(savename)  
      % write 3D optode positions
      fid = fopen([savename '_3D_channel_positions.txt'], 'w');
      % lines: x y z comment
      for i=1:size(pos, 1)
        fprintf(fid, '%.2f %.2f %.2f\n', pos(i, 1), pos(i, 2), pos(i, 3));
      end
      fclose(fid);  
      end
    end

    if isOptoOrder
      if length(nanstart) == 6 % new version with fiducials
        fdpos = data(1:nanstart(1)-1, :);
        rxpos = data(nanstart(2)+1:nanstart(3)-1, :);
        txpos = data(nanstart(4)+1:nanstart(5)-1, :);
      else % old version        
        rxpos = data(1:nanstart(1)-1, :);
        txpos = data(nanstart(2)+1:nanstart(3)-1, :);
      end
      pos = [rxpos; txpos];
      pos(order{1}, :) = pos(order{2}, :);
      if ~isempty(savename)  
      % write 3D optode positions
      fid = fopen([savename '_3D_optode_positions.txt'], 'w');
      % lines: x y z comment
      for i=1:size(pos, 1)
        fprintf(fid, '%.2f %.2f %.2f\n', pos(i, 1), pos(i, 2), pos(i, 3));
      end
      fclose(fid);  
      end
    end
    
  case {'atlasviewer'}
  
    %% prepare local storage of the output -- This should be digpts.txt
    if nargin < 3
      [savename,pathname] = uiputfile('digpts.txt', 'Save Oxysoft MNI-Export as "digpts.txt"');
      savename = fullfile(pathname, savename);
    end
    [q, N, e] = fileparts(savename);

    if ~strcmp(N,'digpts')
      warning('The txt-file has to be named "digpts.txt". Renaming file to "digpts.txt".');
      N = 'digpts';  
      savename = fullfile(q, [N e]);
    end 

    %% Read the xls file 
    
    % Readtable can't be used, differs between matlab versions    
    data = readcell(fullfile(p, n));
    
    %% Reformat
	
    % Check for missing func
    isempty = @(x) any(ismissing(x));

    % Apply to all cells
    emptyrows = any(cellfun(isempty, data), 2);

    % Remove the rows where any cell is missing
    data(emptyrows, :) = [];
    
    % Convert to table
    data = cell2table(data);    
    
    % Remove channel data
    data = data(1:find(strcmp(data.data1, 'Channel')==1)-1,:);
    
    % Remove lines without coordinate data
    fiducial_line = find(strcmp(data.data1, 'Fiducial')==1) ;
    optode_lines = transpose(find(strcmp(data.data1, 'Optode')==1));
    data([fiducial_line, optode_lines],:) = [];
    
    % Change coordinate vars to double
    data.data2 = [data.data2{:}]';
    data.data3 = [data.data3{:}]';
    data.data4 = [data.data4{:}]';
    
    %Change labels to comply with atlasviewer

    old_labels = ["Nz","Iz","RPA","LPA","Cz","Rx","Tx"];
    new_labels = ["nz","iz","a2","a1","cz","d","s"];
    for i = 1:length(old_labels)
        data.data1 = strrep(data.data1, old_labels(i), new_labels(i));
    end

    % Add space and colon
    data.data1 = strcat(data.data1, ' :');
    
    % Round to one decimal
    data(:,[2,3,4]) = array2table(round(data{:,[2,3,4]},1));

    %% De-duplicate s/d, for multi-device measurements
        
    % Extract the 'Label' column from the table
    labels = data.data1;
    maxS = 0; % To track the highest 's' number
    maxD = 0; % To track the highest 'd' number

    % First pass: find the max numbers for s and d
    for i = 1:length(labels)
        label = labels{i};
        baseLabel = label(1); % 's' or 'd'
        numPart = regexp(label, '\d+', 'match'); % Extract numerical part
	if baseLabel == 's' || baseLabel == 'd'
		if ~isempty(numPart)
			num = str2double(numPart{1});
			if baseLabel == 's'
				maxS = max(maxS, num);
			elseif baseLabel == 'd'
				maxD = max(maxD, num);
			end
		end
	end
    end

    % Initialize tracking for duplicates
    seenLabels = containers.Map('KeyType', 'char', 'ValueType', 'int32');

    % Second pass: rename duplicates by adding max number
    for i = 1:length(labels)
        label = labels{i};
        baseLabel = label(1); % 's' or 'd'
        numPart = regexp(label, '\d+', 'match'); % Extract numerical part again
		if baseLabel == 's' || baseLabel == 'd'		
			if ~isempty(numPart)
				num = str2double(numPart{1});
				originalLabel = sprintf('%s%d', baseLabel, num); % Reconstruct original label
	
				if isKey(seenLabels, originalLabel)
					% Duplicate found, add max number accordingly
					if baseLabel == 's'
						num = num + maxS;
					elseif baseLabel == 'd'
						num = num + maxD;
					end
					% Generate new label with updated number
					newLabel = sprintf('%s%d%s', baseLabel, num, ' :');
				else
					% No duplicate, just use original label
     					seenLabels(originalLabel) = 1; % Mark this label as seen
					newLabel = sprintf('%s%s', originalLabel, ' :');
					
				end
	
				% Update the label in the table
				labels{i} = newLabel;
			end
		end
    end
            
    % Update data with new labels
    data.data1 = labels;

    
    %% Write output       
    if ~isempty(savename)  
        writetable(data, savename, 'Delimiter', ' ','WriteVariableNames', false, 'QuoteStrings', false);
    end

  case {'nirstorm'}     
        
    [savename,pathname] = uiputfile('fiducials.txt', 'Save Oxysoft MNI-Export as "fiducials.txt"');
    savename = fullfile(pathname, savename);
    [savename2,pathname2] = uiputfile('optodes.txt', 'Save Oxysoft MNI-Export as "optodes.txt"');
    savename2 = fullfile(pathname2, savename2);
    
    [q, N, e] = fileparts(savename);
    [q2, N2, e2] = fileparts(savename2);

    if ~strcmp(N,'fiducials')
      warning('The txt-file has to be named "fiducials.txt". Renaming file to "fiducials.txt".');
      N = 'fiducials';  
      savename = fullfile(q, [N e]);
     end  
        
    if ~strcmp(N2,'optodes')
      warning('The txt-file has to be named "optodes.txt". Renaming file to "optodes.txt".');
      N2 = 'optodes';  
      savename2 = fullfile(q2, [N2 e2]);
    end  
              
       
    %% Read the xls file
    data = xlsread(fullfile(p, n));
    nanstart = find(isnan(data(:, 1)));   
      
    %% Extract the positions, reorder and save
    Fiducial = data(1:nanstart(1)-1,:);
    Fiducial_reorder = [Fiducial(1,:); Fiducial(4,:); Fiducial(3,:); Fiducial(5,:); Fiducial(2,:)];
    pos_names  = ['Nasion  ';'LeftEar ';'RightEar';'CZ      ';'IZ      '];

    % Receivers/detectors:
    pos_Rx = data(nanstart(2)+1:nanstart(3)-1,:);
    nrR = nanstart(3)-1 - nanstart(2);  % number of receivers

    % Transmitters/sources
    pos_Tx = data(nanstart(4) +1: nanstart(5)-1,:);
    nrT = nanstart(5)-1 - nanstart(4);  % number of transmitters

    if nrT > 9
      % apparently, there needs to be a trailing white space for idx < 10
      Tarray1(1:9,:) = [char(ones(9,1) * 'S'), num2str([1:9]'), char(ones(9,1) * ' ')]; 
      diff  = nrT-9;
      Tarray2(1:diff,:) = [char(ones(diff,1) * 'S'), num2str([10:nrT]')];
      Tarray = [Tarray1; Tarray2];
    else
      Tarray = [char(ones(nrT,1) * 'S') ,num2str([1:nrT]')];
    end

    if nrR > 9
      % apparently, there needs to be a trailing white space for idx < 10
      Rarray1(1:9,:) = [char(ones(9,1) * 'D') ,num2str([1:9]'), char(ones(9,1) * ' ')];
      diff  = nrR-9;
      Rarray2(1:diff,:) = [char(ones(diff,1) * 'D') ,num2str([10:nrR]')];
      Rarray = [Rarray1;Rarray2];
    else
      Rarray = [char(ones(nrR,1) * 'D') ,num2str([1:nrR]')];
    end

    %% determine max number of characters of name of transmitter/detector
        
    for l = 1: length(Tarray); 
      L2 = max(length(Tarray(l,:)));
    end 
    for m = 1: length(Rarray); 
      L3 = max(length(Rarray(m,:)));
    end

    L = max([L2,L3]);    %determine the max. length

    if L2<L 
      for l = 1:length(Tarray); 
      X(l,:) = char(strcat(Tarray(l,:), {' '}));
      end
      Tarray = X;
    end
    if L3<L 
      for m = 1:length(Rarray); 
      X(m,:) = char(strcat(Rarray(m,:), {' '}));
      end
      Rarray = X;
    end

    %% put optodes together
    
    o_index = [Tarray; Rarray];
    o_pos = [pos_Tx; pos_Rx]; 

    %% write fiducials.txt
    if ~isempty(savename)  
      fid = fopen([savename], 'w');
      for i=1:size(pos_names, 1)
        fprintf(fid, '%s\t1\t11\t%.1f\t%.1f\t%.1f\t0\r\n', strtrim(pos_names(i,:)), Fiducial_reorder(i, 1), Fiducial_reorder(i, 2), Fiducial_reorder(i, 3));
      end
      fclose(fid);   
    end      
    %% write optodes.txt
    if ~isempty(savename2)  
      fid = fopen([savename2], 'w');
      for i=1:size(o_index, 1)
        fprintf(fid, '%s\t1\t11\t%.1f\t%.1f\t%.1f\t0\r\n', strtrim(o_index(i,:)), o_pos(i, 1), o_pos(i, 2), o_pos(i, 3));
      end
      fclose(fid);  
    end

  end % switch
  
end %eof
