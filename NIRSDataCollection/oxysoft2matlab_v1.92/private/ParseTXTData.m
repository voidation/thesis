function [OD, metaInfo, ADValues] = parseTXTData(filename)
% use as
%   [OD, metaInfo, ADValues] = parseTXTData(filename)
% output organized as in oxy3read_function_v3
% See also OXY3READ_FUNCTION_V3 

fileid = fopen(filename);

if fileid==-1
    error('file not found or cannot be opened.');
end

while(true)
    line = fgetl(fileid);
    
    string = 'Datafile sample rate:';
    begIdx = strfind(line, string);    
    if isempty(begIdx)
        endIdx = strfind(line, 'Hz');
        metaInfo.SampleTime = 1/str2double(line(begIdx+numel(string):endIdx-1));
    end
    
    string = 'Datafile total number of samples:';
    begIdx = strfind(line, string);    
    if isempty(begIdx)
        endIdx = numel(line);
        metaInfo.nbSamples = str2double(line(begIdx+numel(string):endIdx));
    end
    
    string = 'Optode distance (mm):';
    begIdx = strfind(line, string);    
    if isempty(begIdx)
        endIdx = numel(line);
        metaInfo.Position = str2double(line(begIdx+numel(string):endIdx));
    end
    
    string = 'Optode distance (mm):';
    begIdx = strfind(line, string);    
    if isempty(begIdx)
        endIdx = numel(line);
        metaInfo.fs = str2double(line(begIdx+numel(string):endIdx));
    end

keyboard

end