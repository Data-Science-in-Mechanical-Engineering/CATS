%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

1;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% workaround for uiputfile(), which does not work with Octave 4.4.1 on MacOS
if (ismac())
function [fname, fpath, fltidx] = uiputfile(a, b = [])

	while (1)
		
		[fname, fpath, fltidx] = uigetfile(a,b);

		if (fname)
			f = fopen([fpath, fname], "r");
			if (f >= 0)
				fclose(f);
				if (!strcmp("Yes", questdlg([fname " already exists.\nDo you want to replace it?"], b, "Yes", "No", "No")))
					continue;
				endif
			endif
		endif	

		break;
		
	endwhile
	
endfunction
endif

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [logData, logDetails] = runTestRound(testbed, initiator, logLevel)

	global nodes;
	global config;

	Pnoise = 10 ^ (testbed.linkModel.Pnoise_dBm / 10);

	logFlags = struct();
	logFlags.packets			= (bitget(logLevel, 32) == 0);
	if (isargout(2))
		logFlags.packetDetails 	= (bitget(logLevel,  1) != 0);
		logFlags.nodes   		= (bitget(logLevel,  2) != 0);
	else
		logFlags.packetDetails	= false;
		logFlags.nodes   		= false;
	endif

	logData.rank 			= uint8(zeros(config.numSlots, length(nodes)));
	logData.packetSource 	= uint8(zeros(size(logData.rank)));

	logDetails = [];

	for i = 1 : length(nodes)
		init(i, initiator);
	endfor

	if (logFlags.packets)
		logData.packetCoeffs = uint8(zeros(length(nodes), columns(nodes(1).coeffs), config.numSlots));
		if (config.request.mode)
			logData.packetRequest = uint8(zeros(length(nodes), columns(nodes(1).coeffs), config.numSlots));
		endif
	endif

	if (logFlags.packetDetails)
		logDetails(1).packets = [];
	endif
	if (logFlags.nodes)
		logDetails(1).nodes = nodes;
	endif

	for slot = 1 : config.numSlots
%slot, fflush(stdout);
		% first slot has index 1 (also in the real implementation),
		% birthSlot = 0 marks an empty row

		packets = cell(length(nodes), 1);

		txNodes = [];
		links = [];
		
		for k = 1 : length(nodes)
			p = transmit(k, slot);
			if (!isempty(p))
				txNodes(end+1) 	= k;
				packets{k}		= p;
				links(:,end+1) 	= testbed.linkMatrix_dB(:,k);
				if (logFlags.packets)
					logData.packetCoeffs(k,:,slot) = p.coeffs;
					if (config.request.mode)
						logData.packetRequest(k,:,slot) = bitand(p.flags, 3) * p.requestField;
					endif
				endif
			endif
		endfor
%toc()
%txNodes
		rxNodes = setdiff([1 : length(nodes)], txNodes);

		rxMap = zeros(1, length(nodes));
		if (!isempty(txNodes))
			for k = rxNodes
				% TODO: allow randomized Tx power per node (i.e. continues/multivalued scale instead of on/off)
				l = 10 .^ ((testbed.linkModel.Ptx_dBm + links(k,:)) ./ 10);

				% selecting i based on max(l) is equivalent to selecting i based on max(p)
				% since SINR=f(l) and p=f(SINR) are both monotonically increasing
				%SINR = l ./ (sum(l) - l + testbed.linkModel.Pnoise);
				%SINR_dB = 10 * log10(SINR);
				%p = testbed.linkModel.SINR2p(SINR_dB);
				%[p,i] = max(p);
				[~,i] = max(l);
				SINR = l(i) ./ (sum(l) - l(i) + Pnoise);
				SINR_dB = 10 * log10(SINR);
				p = testbed.linkModel.SINR2p(SINR_dB);
				if (rand() < p)
					i = txNodes(i);
					rxMap(k) = i;
					packets{k} = packets{i};
					if (logFlags.packets)
						logData.packetCoeffs(k,:,slot) = packets{k}.coeffs;
						if (config.request.mode)
							logData.packetRequest(k,:,slot) = bitand(packets{k}.flags, 3) * packets{k}.requestField;
						endif
					endif
				endif
			endfor
		endif
%toc()
%rxMap
		for k = rxNodes
			receive(k, packets{k}, slot);
		endfor
%toc()
		logData.rank(slot,:) 				= [nodes.rank];
		logData.packetSource(slot,:) 		= rxMap;
		logData.packetSource(slot,txNodes) 	= txNodes;

		if (logFlags.packetDetails)
			logDetails(slot).packets = packets;
		endif
		if (logFlags.nodes)
			logDetails(slot).nodes = nodes;
		endif
%toc()

%		% check if payload is correct at all nodes
%		p = [];
%		for k = 1 : length(nodes)
%			p(k,:) = GFunpack(uint8(k), config.fieldSize)(end:-1:1);
%		endfor
%		for n = nodes
%			for k = 1 : rows(n.coeffs)
%				x = uint8(zeros(1, columns(p)));
%				for l = 1 : rows(n.coeffs)
%					x = config.GFadd(x, config.GFmul(n.coeffs(k,l), p(l,:)));
%				endfor
%				if (!isequal(x, n.payloads(k,1 : length(x))))
%					warning(["corrupted payload detected: slot %d node %d row %d: " ...
%						"expected = (%s\b), seen = (%s\b)"], ...
%						slot, n.id, k, sprintf("%d ", x), sprintf("%d ", n.payloads(k,1:length(x))));
%error("stop");						
%				endif	
%			endfor
%		endfor

	endfor	

	% check if decoded payload is correct at all finished nodes
%	if (mixerVersion...)
		r = 1 : length(GFunpack(uint8(0), config.fieldSize));
		for n = nodes
			if (n.rank == rows(n.coeffs))
				for k = 1 : rows(n.coeffs)
					p = GFpack(n.payloads(k,r)(end:-1:1), config.fieldSize, "uint8");
					if (p != k)
						warning(["corrupted payload detected: node %d row %d: " ...
							"expected = %d, decoded = %d (%s\b)"], ...
							n.id, k, k, p, sprintf("%d ", n.payloads(k,:)));
					endif	
				endfor
			endif
		endfor
%	endif

endfunction

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [logData, logDetails] = replayTestRound(testbed, round_, logLevel)

	global nodes;
	global config;


	% extract trace of round <round_>

	[~, I] = sort(config.traceMap(:, round_));
	dataMap = config.traceMap(I, round_ : round_ + 1);

	f = fopen(config.traceFile, "rb");

	dataString = "";
	for i = 1 : length(nodes)
		if (dataMap(i,1) >= ftell(f))
			fseek(f, dataMap(i,1), SEEK_SET);
			dataString = [dataString, fread(f, [1, dataMap(i,2) - dataMap(i,1)], "*char")];
		elseif (dataMap(i,2) > ftell(f))
			dataString = [dataString, fread(f, [1, dataMap(i,2) - ftell(f)], "*char")];
		endif
		dataMap(i,2) -= dataMap(i,1);
		dataMap(i,1) = 1 + length(dataString) - (ftell(f) - dataMap(i,1));
		dataMap(i,2) += dataMap(i,1) - 1;
	endfor

	fclose(f);


	% prepare log data structures, init nodes

	logFlags = struct();
	logFlags.packets			= (bitget(logLevel, 32) == 0);
	if (isargout(2))
		logFlags.packetDetails 	= (bitget(logLevel,  1) != 0);
		logFlags.nodes   		= (bitget(logLevel,  2) != 0);
	else
		logFlags.packetDetails	= false;
		logFlags.nodes   		= false;
	endif

	logData.rank 			= uint8(zeros(config.numSlots, length(nodes)));
	logData.packetSource 	= uint8(zeros(size(logData.rank)));
	if (logFlags.packets)
		logData.packetCoeffs = uint8(zeros(length(nodes), length(nodes), config.numSlots));
		if (config.request.mode)
			logData.packetRequest = uint8(zeros(length(nodes), length(nodes), config.numSlots));
		endif
	endif

	logDetails = [];

	for i = 1 : length(nodes)
		init(i, 0);
	endfor

	if (logFlags.packetDetails)
		logDetails(1).packets = [];
	endif
	if (logFlags.nodes)
		logDetails(1).nodes = nodes;
	endif


	% read packets from trace

	patternBitfieldLength = 2 * fix((length(nodes) + 7) / 8);
	patternCoeffsLength = patternBitfieldLength;				% TODO: abh�ngig von fieldSize
	pattern = ...
	[
		"([0-9a-f]{4}) - "											... % slotNumber
		"([0-9a-f]{4}) - "											... % senderId
		"([0-9a-f]{2}) - "											... % flags
		"([0-9a-f]{" sprintf("%d", patternBitfieldLength) ",}) - "	... % requestField
		"([0-9a-f]{" sprintf("%d", patternCoeffsLength) ",}) - "	... % coeffs
		"([0-9a-f]{2,})"											... % payload
	];

	packets = cell(length(nodes), config.numSlots + 1);
	isTx = logical(zeros(length(nodes), config.numSlots + 1));

	for l = 1 : length(nodes)

		k = I(l);
		s = regexp(dataString(dataMap(l,1) : dataMap(l,2)), [config.fLineHeader(sprintf("%d", testbed.nodes(k,2))) pattern], "tokens");

		for i = 1 : length(s)

			n = uint16(hex2dec(s{i}(1:3)));
			slot = n(1);

			% note: first slotNumber is 1, 0 is used to mark broken packets
			if (slot < 1)
				continue;
			elseif (slot > config.numSlots)
				error("trace data is inconsistent\n");
			endif

			p = struct("source", uint8(bitand(n(2),0x00ff))+1, "flags", uint8(n(3)));

			% senderId's MSB marks vector bit order:
			% 0: LSB first, big-endian
			% 1: LSB first, little-endian
			% -: MSB first, big-endian (natural for human reading)			
			if (n(2) >= 0x8000)
				t = postpad(s{i}{5}, patternCoeffsLength, "0");
				t = uint8(hex2dec(reshape(t, 2, [])'))';
				p.coeffs = bitunpack(t)(1 : length(nodes));
%				t = postpad(s{i}{5}, patternCoeffsLength, "0");
%				t = fliplr(uint8(hex2dec(reshape(t, 2, [])'))');
%				p.coeffs = fliplr(bitunpack(t)(end - length(nodes) + 1 : end));
			else
				t = prepad(s{i}{5}, patternCoeffsLength, "0");
				t = fliplr(uint8(hex2dec(reshape(t, 2, [])'))');
				p.coeffs = bitunpack(t)(1:length(nodes));		% TODO: abh�ngig von fieldSize
			endif

			t = postpad(s{i}{6}, 2 * config.payloadSize, "0");
			t = fliplr(uint8(hex2dec(reshape(t, 2, [])'))');
			p.payload = fliplr(bitunpack(t));

			if (config.request.mode)
				if (n(2) >= 0x8000)
					t = postpad(s{i}{4}, patternBitfieldLength, "0");
					t = uint8(hex2dec(reshape(t, 2, [])'))';
					p.requestField = bitunpack(t)(1 : length(nodes));
				else
					t = prepad(s{i}{4}, patternBitfieldLength, "0");
					t = fliplr(uint8(hex2dec(reshape(t, 2, [])'))');
					p.requestField = bitunpack(t)(1:length(nodes));
				endif
			endif

			packets{k,slot} = p;
			if (p.source == k)
				isTx(k,slot) = true;
			endif

			logData.packetSource(slot,k) = p.source;
			if (logFlags.packets)
				logData.packetCoeffs(k,:,slot) = p.coeffs;
				if (config.request.mode)
					logData.packetRequest(k,:,slot) = bitand(p.flags, 3) * p.requestField;
				endif
			endif

			% TODO: falls das zu langsam ist: nur Tx-Pakete parsen und Rx-Pakete mit String-Vergleich zuordnen
			% -> kann auch �bertragungsfehler aufdecken

		endfor

	endfor


	% replay packets
	for slot = 1 : config.numSlots

		for k = 1 : length(nodes)
			if (isTx(k, slot + 1))
				receive(k, packets{k,slot}, slot, isTx(k,slot), packets{k, slot + 1});
			else
				receive(k, packets{k,slot}, slot, isTx(k,slot), []);
			endif
		endfor

		logData.rank(slot,:) = [nodes.rank];

		if (logFlags.packetDetails)
			logDetails(slot).packets = packets(:,slot);
		endif
		if (logFlags.nodes)
			logDetails(slot).nodes = nodes;
		endif

	endfor	

endfunction

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function logData = runTests(testcase, logFile)

	global nodes;
	global config;
	global mixerVersion;
%tic();
	logData = [];
	logPackets = true;

	if (!isempty(logFile))
		save('-text', '-zip', logFile, 'testcase');
	else
		switch (testcase.logDetailsMode)
			case 'logData'
			otherwise
				testcase.logDetailsMode = 'none';
		endswitch
	endif

	if (isequal(testcase.logDetailsMode, 'none'))
		testcase.logLevel = uint32(0x80000000);
		logPackets = false;
	endif

	if (!isfield(testcase, "simulationMode"))
		testcase.simulationMode = "default";
	endif

	for t = 1 : length(testcase.testbeds)

		if (isstruct(testcase.testbeds))
			testbed = testcase.testbeds(t);
			testbedLabel = sprintf("testbed %d", t);
		else
			load(testcase.testbeds{t}, 'testbed');
			testbedLabel = testcase.testbeds{t};
		endif

		nodes = struct('id', num2cell([1 : rows(testbed.nodes)]));

		for c = 1 : length(testcase.config)

			config = testcase.config(c);

			config.simulationMode = testcase.simulationMode;
			if (!isfield(config, "rounds"))
				config.rounds = 1 : config.numRounds;
			endif

			if (isstruct(testcase.testbeds))
				logData(end+1).testbed 	= testcase.testbeds(t);
			else
				logData(end+1).testbed 	= testcase.testbeds{t};
			endif
			logData(end).config 		= config;
			logData(end).mixerVersion	= mixerVersion;
			logData(end).rank 			= uint8(zeros(config.numSlots, length(nodes), config.numRounds));
			logData(end).packetSource 	= uint8(zeros(size(logData(1).rank)));
			logData(end).RNGSeed 		= zeros(625, config.numRounds);
			logData(end).startTime		= time();

			% work with some local variables because access to logData(end)... is very slow for some reason
			if (logPackets)
				packetCoeffs = [];
			endif
%tic();
			for roundIndex = 1 : length(config.rounds)

				round_ = config.rounds(roundIndex);

				printf("processing %s config %d round %-5d\t(%s)\n", testbedLabel, c, round_, datestr(now(), 13)); fflush(stdout);

				logData(end).RNGSeed(:,round_) = rand('state');
%toc()
				switch (testcase.simulationMode)
					case "replayTrace"
						[logRound, logDetails] = replayTestRound(testbed, round_, testcase.logLevel);
					otherwise
						initiator = config.fInitiator(round_, rows(testbed.nodes));
						[logRound, logDetails] = runTestRound(testbed, initiator, testcase.logLevel);
				endswitch
%toc()
				logData(end).rank(:,:,roundIndex) 			= logRound.rank;
				logData(end).packetSource(:,:,roundIndex) 	= logRound.packetSource;
				if (logPackets)
					if (isempty(packetCoeffs))
						packetCoeffs = uint8(zeros([size(logRound.packetCoeffs), config.numRounds]));
						if (config.request.mode)
							packetRequest = uint8(zeros([size(logRound.packetRequest), config.numRounds]));
						endif
					endif
					packetCoeffs(:,:,:,roundIndex)			= logRound.packetCoeffs;
					if (config.request.mode)
						packetRequest(:,:,:,roundIndex)		= logRound.packetRequest;
					endif
				endif

				if (!isempty(logDetails))
					logName = sprintf('logDetails_%d_%d_%d', t, c, round_);
					switch (testcase.logDetailsMode)
						case 'logData'
							logData(end).(logName) = logDetails;
						case 'logFile'
							temp = struct(logName, logDetails);
							save('-append', '-text', '-zip', logFile, '-struct', 'temp');
						case 'extraFiles'
							[dir, name, ext] = fileparts(logFile);
							name = fullfile(dir, [name '_' logName ext]);
							temp = struct(logName, logDetails, [logName '_timeStamp'], time());
							save('-text', '-zip', name, '-struct', 'temp');
						otherwise
							logName = [];
					endswitch
					logData(end).logDetailsName{roundIndex} = logName;
				endif
%toc()
			endfor

			if (logPackets)
				logData(end).packetCoeffs = packetCoeffs;
				if (config.request.mode)
					logData(end).packetRequest = packetRequest;
				endif
			endif

			logData(end).stopTime = time();

		endfor
	endfor

	if (!isempty(logFile))
		printf('writing log data to %s...\n', logFile); fflush(stdout);
		save('-append', '-text', '-zip', logFile, 'logData');
	endif

endfunction

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [fLineHeader, testbed, setup, rounds, traceMap] = scanTraceFile(fileName)

	fLineHeader	= [];
	testbed 	= struct();
	setup		= struct();
	rounds		= [];
	traceMap	= [];

	f = fopen(fileName, "rb");

	fseek(f, 0, SEEK_END);
	len = ftell(f);
	fseek(f, 0, SEEK_SET);
	if (len > 100 * 1024^2)
		b = questdlg( ...
				[sprintf("File size is quite large (%d MB),\n", round(len / 1024^2))	...
				"processing may be slow and memory consuming.\nContinue?"], ...
				"Warning", "Cancel", "Ok", "Cancel");
		if (!strcmp(b, "Ok"))
			return;
		endif
	endif

	s = fread(f, [1, inf], "*char");
	fclose(f);

	% detect trace file format
	p = {"(\\d+)"; "[^\\n]*SETUP:([^\\n]*)"; "EXP ([^\\n]*)"};
	fmt = 1;
  if (1)
		% common log format
		%fLineHeader = @(s) ["\\d+\\.\\d+\\s*\\|\\s*\\d+\\s*\\|" s];
		fLineHeader = @(s) ["\\d+\\.\\d+\\s*\\|\\s*" s "\\s*\\|\\s*(?:#\\s*ID:\\d+)?\\s*"];
		s1 = regexp(s, [fLineHeader("\\s*\\d+\\s*")], "matches", "once");
    fmt = 3
	endif
	if (isempty(s1))
		% desktop, Cooja
		fLineHeader = @(s) ["\\S+\\s+ID:" s "\\s+"];
		s1 = regexp(s, [fLineHeader(p{1}) p{2}], "matches", "once");
    %disp(fLineHeader(p{1})) "\S+\s+ID:(\d+)\s+[^\\n]*SETUP:([^\\n]*)"
	endif
	if (isempty(s1))
		% FlockLab
		fLineHeader = @(s) ["[.\\d]+,\\d+," s ",r,"];
		s1 = regexp(s, [fLineHeader(p{1}) p{2}], "matches", "once");
		if (isempty(s1))
			s1 = regexp(s, [fLineHeader(p{1}) p{3}], "matches", "once");
			if (!isempty(s1))
				fmt = 2;
			endif
		endif
	endif
	if (isempty(s1))
		% Indriya (preprocessed)
		fLineHeader = @(s) ["\\d+," s ","];
		s1 = regexp(s, [fLineHeader(p{1}) p{2}], "matches", "once");
	endif
	if (isempty(s1))
		error("unsupported trace file format\n");
	endif

	% detect nodes
	switch (fmt)
		case 1
			s1 = regexp(s, [fLineHeader("(?<id>\\d+)") "[^\n]*SETUP:(?<setup>[^\n]*)"], "names");
		case 2
			s1 = regexp(s, [fLineHeader("(?<id>\\d+)") "EXP(?<setup>[^\n]*)"], "names");
    case 3
			%s1 = regexp(s, [fLineHeader("(?<id>\\d+)") ".*\\n"], "names")
      s1 = regexp(s, [fLineHeader("(\\d+)") "\\s*starting node (?<id>\\d+)"], "names")
      s2 = regexp(s, [fLineHeader("(\\d+)") "\\s*MX_NUM_NODES\\s*= (?<nodes>\\d+)"], "names", "once")
      s3 = regexp(s, [fLineHeader("(\\d+)") "\\s*MX_PAYLOAD_SIZE\\s*= (?<payload>\\d+)"], "names", "once")
      s4 = regexp(s, [fLineHeader("(\\d+)") "\\s*MX_ROUND_LENGTH\\s*= (?<slots>\\d+)"], "names", "once")
      %s3 = regexp(s, [fLineHeader("(?<id>\\d+)") ".*\\n"], "names")
	endswitch
	nodes = sort(str2double({s1.id}));
%	nodes = sort(str2double(s1.id));
	testbed.nodes = [1 : length(nodes); nodes]';
	testbed.area = [length(nodes) + 1, max(nodes) + 1];
	nodesMap = [];
	nodesMap(nodes) = 1 : length(nodes);

%{
	% read setup (from first node)
	s1 = regexp(s1(1).setup, "(?<name>\\w+) ?= ?(?<value>[^, ]+)", "names");
%	s1 = regexp(s1.setup{1}, "(?<name>\\w+) ?= ?(?<value>[^, ]+)", "names");
	for i = 1 : length({s1.value})
		s1(i).value = eval(s1(i).value);
%	for i = 1 : length(s1.value)
%		s1.value{i} = eval(s1.value{i});
  endfor
%}

  for i = 1 : length({s1.id})
    s1(i).nodes = eval(s2.nodes);
    s1(i).payload = eval(s3.payload);
    s1(i).slots = eval(s4.slots);
  endfor

	%setup = cell2struct({s1.value}, {s1.name}, 2);
  setup = cell2struct({eval(s2.nodes), eval(s3.payload), eval(s4.slots)}, {"nodes", "payload", "slots"}, 2)
%	setup = cell2struct(s1.value, s1.name, 2);
	if (!isfield(setup, "slots"))
		setup.slots = setup.roundLen;
	endif
	switch (fmt)
		case 1
			setup.fmtINFOS = "INFOS \\((?<round>\\d+)\\)";
		case 2
			setup.fmtINFOS = "RND (?<round>\\d+)";
    case 3
      %setup.fmtINFOS = "\\s*# ID:\\s*\\d+\\s*(?<round>\\d+)";
      setup.fmtINFOS = "starting round (?<round>\\d+)";
	endswitch
  
	if (setup.nodes != length(nodes))
		warning("number of nodes is inconsistent (setup: %u, found: %u)\n", setup.nodes, length(nodes));
	endif

	% scan rounds and trace data areas
	traceMap = [];
  [s1,p] = regexp(s, [fLineHeader("(?<id>\\d+)") setup.fmtINFOS], "names", "start");
%	[s1,p] = regexp(s, [fLineHeader("(?<id>\\d+)") setup.fmtINFOS], "names", "start");
%	[s1,p] = regexp(s, [fLineHeader("(?<id>\\d+)") "INFOS \\((?<round>\\d+)\\)"], "names", "start");
%	[s1,p] = regexp(s, [fLineHeader("(?<id>\\d+)") "RND (?<round>\\d+)"], "names", "start");
	for i = 1 : length(p)
		traceMap(nodesMap(str2double(s1(i).id)), str2double(s1(i).round)) = p(i) - 1;
%		traceMap(nodesMap(str2double(s1.id{i})), str2double(s1.round{i})) = p(i) - 1;
	endfor

	rounds = 1 : columns(traceMap);
	if (!isempty(rounds))
		while (max(traceMap(:,rounds(1))) == 0)
			rounds(1) = [];
		endwhile
		traceMap(find(traceMap == 0)) = len;
		traceMap(:, end+1) = len;
	endif

	% drop trace data since it may be large
	s = [];

endfunction

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function plotRankVsNode(rank)

	numNodes = size(rank, 2);
	numSlots = size(rank, 1);

	rank = [ones(1, numNodes); rank; rank(end,:)];
	rank = [rank, rank(:,end)]';

	X = [0 : numSlots + 1] - 0.5;
	Y = [0 : numNodes] + 0.5;
%	pcolor(X, Y, rank);
	surf(X, Y, rank);
	view(2);
	axis([X(1), X(end), Y(1), Y(end)]);
	caxis([0 numNodes]);
	colormap('jet');
%	colormap(jet()(end:-1:1,:));
	colorbar();

	xlabel('slot');
	ylabel('node');

endfunction

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

Mixer;

global nodes;
global config;

action = menu('choose action:', ...
				'generate testcase',	...
				'load testcase', 		...
				'run test(s)',			...
				'replay trace file',	...
				'import log file',		...
				'push results',			...
				'pop results',			...
				'evaluate results' 		...
			);
switch (action)

	% generate testcase
	case 1
		if (exist('testcase') && length(testcase.paramStrings) == 6)
			params = testcase.paramStrings;
		else	
			params = ...
			{	'sprintf("data/testbed_%d.dat;", [1:1])',
				'10',
				'50',
				'4',
%				'',
				'0b11',
%				'sprintf("config %d;", [1:1])',
				'0'
			};
		endif

		params = inputdlg( ...
			{	'testbed(s)',
				'numRounds',
				'numSlots',
				'payloadSize',
%				'other options (key=val;key=val;...)',
				'log level (flags: 1 = incl. packets, 2 = incl. node states)',
%				'config label(s)',
				'save details in (0 = drop, 1 = log structure, 2 = log file, 3 = extra files)'
			}, ...
			'testcase parameters', 1, ...
			params ...
		);

		if (length(params) > 0)

			testcase.paramStrings = params;
			testcase.simulationMode = "default";

			testcase.testbeds = strtrim(strsplit(eval(params{1}), ';'));
			if (isempty(testcase.testbeds{end}))
				testcase.testbeds(end) = [];
			endif

%			s = ["'numRounds'," params{2} ",'numSlots'," params{3} ",'payloadSize'," params{4}];
%			if (length(params{5}) > 0)
%				o = strsplit(params{5}, {';','='});
%				s = [s sprintf(",\"%s\",%s", o{:})];
%			endif
%			testcase.config = eval(['struct(' s ');']);

			if (isfield(testcase, 'configSource'))
				configSource = strrep(strrep(testcase.configSource, "@n", "\n"), "@@", "@");
			else	
				configSource = [ ...
					"i=1;																			\n" ...
					"config(i).label = sprintf(\"config %d\", i);									\n" ...
					"config(i).fieldSize = 2;														\n" ...
					"config(i).payloadDistribution = '[1 : length(nodes)]';							\n" ...
					"config(i).fInitiator = @(r,n) 1; %1 + rem(r - 1, n);							\n" ...
					"config(i).fTimeout = @(n) 5 + randi(5) - 4;									\n" ...
					"\t% nominal timeout: 5 +/- 2 = 3...7 slots from now							\n" ...
					"config(i).historyWindowLength = 'round([3, 1] * length(nodes))';				\n" ...
					"config(i).exchangeTriggerSparsity = 'length(nodes)';							\n" ...
					"config(i).immediateElimination = false;										\n" ...
					"config(i).fastTxUpdate.allowedUpdates = 1;										\n" ...
					"config(i).fastTxUpdate.mulOnRx = true;											\n" ...
					"config(i).fastTxUpdate.mulOnRequest = true;									\n" ...
					"config(i).fastTxUpdate.mulOnOwn = true;										\n" ...
					"config(i).smartShutdown = true;												\n" ...
					"config(i).recursiveNeighborhood = 1;											\n"	...
					"config(i).txPacket.includeOwn = 3;												\n" ...
					"\t% -1 = always, 0 = unenforced, x>0 = until x times received (= acked)		\n" ...
					"config(i).txPacket.fAgeToP = @(a) 0.5 + (1 - 0.5) * 2 ^ (-0.5 * a);			\n" ...
					"config(i).txPacket.emptyPacketStrategy = 'own';								\n" ...
					"\t% what to insert: 'own' = own, 'first' = first found; 'random' = random one	\n" ...
					"config(i).fTxCurve = @(a,d,n,dn) 1 ./ (d + 1) + d ./ (d + 1) .* 2 .^ (-0.5 * a);	\n"	...
					"%dr_ = @(d,dn) 1 / (mean([d dn]) + 1);											\n"	...
					"%dr_ = @(d,dn) 1 / (mean([d, min(dn)]) + 1);									\n"	...
					"%dr_ = @(d,dn) 1 / (norm(double([d dn]), 0.5) / length([d dn]) ^ 2 + 1);		\n"	...
					"%config(i).fTxCurve = @(a,d,n,dn) dr_(d,dn) + (1 - dr_(d,dn)) .* 2 .^ (-0.5 * a);	\n"	...
					"config(i).coordinatedSlotting.on = true;										\n"	...
					"config(i).coordinatedSlotting.fpOwn = @(p,d,n,h) 1 - h;						\n" ...
					"config(i).coordinatedSlotting.fpForeign = @(p,d,n,h) h / (d + 1);				\n" ...
					"config(i).coordinatedSlotting.fpInit = @(p,n) 1 / n;							\n" ...
					"config(i).request.mode = 'column,pivot';										\n" ...
					"config(i).request.columnSearchMode = 'pivot';									\n" ...
					"\t% 'pivot' or 'all'															\n" ...
					"config(i).request.rxSnoop = true;												\n" ...
					"config(i).request.fTxColumnYesNo = @(a,r,n) a > (n - r);						\n" ...
					"config(i).request.fTxPivotYesNo = @(a,r,n) a > (n - r);						\n" ...
					"config(i).request.fTxSelect = @(m) rand() < nnz(m) / length(m);				\n" ...
					"config(i).request.fpHelpless = @(a,d,n) 1 / n;									\n" ...
				];
				configSource = strjoin(strtrim(strsplit(configSource, "\n")), "\n");
			endif
			
			configFile = [tempname("", "oct-testcase-config-") ".m"];
			
			f = fopen(configFile, "w");
			fputs(f, configSource);
			fclose(f);

			system([program_invocation_name " --gui --persist --eval \"edit '" configFile "'\""], false, "sync");
%			system(configFile, false, "sync");
			b = questdlg("Ready to continue?", "Edit config options", "Cancel", "Yes", "Yes");
			if (strcmp(b, "Yes"))

				configSource = fileread(configFile);
				configSource = strrep(strrep(configSource, "\r\n", "\n"), "\r", "\n");
				configSource = strrep(strrep(configSource, "@", "@@"), "\n", "\n@");
				configSource = strjoin(strtrim(strsplit(configSource, "\n", "collapsedelimiters", false)), "\n");
				testcase.configSource = strrep(configSource, "\n@", "@n");
				% "\n@" is temporarily used to avoid that strtrim() trims something on the left (like tabs)

				configTemp = config;
				config = [];
%				clear config;
				source(configFile);
				testcase.config = config;
				config = configTemp;
				clear("configTemp");

				for i = 1 : length(testcase.config)
					testcase.config(i).numRounds = str2num(params{2});
					testcase.config(i).numSlots = str2num(params{3});
					testcase.config(i).payloadSize = str2num(params{4});
				endfor	
				
%				labels = strtrim(strsplit(eval(params{7}), ';'));
%				[testcase.config.label] = labels{1 : length(testcase.config)};

				testcase.logLevel = uint32(eval(params{5}));
				testcase.logDetailsMode = {'none', 'logData', 'logFile', 'extraFiles'}{eval(params{6}) + 1};
				
				[fileName, path] = uiputfile('data/testcase.tcd', 'save testcase as');
				if (fileName)
					save([path fileName], 'testcase');
				endif
				
			endif
			
			delete(configFile);
			
		endif

	% load testcase
	case 2
		[fileName, path] = uigetfile({'*tcd;*.dat', 'Testcase files'}, 'Load testcase', 'data/');
		if (ischar(fileName))
			load([path fileName], 'testcase');
		endif

	% run test(s)
	case 3
		[files, path] = uigetfile({'*tcd;*.dat', 'Testcase files'}, 'Select testcase(s)', 'data/', 'MultiSelect', 'on');
		if (ischar(path))

			if (!iscell(files))
				testcaseFiles = fullfile(path, {files});
				[~, name] = fileparts(files);
				[files, path] = uiputfile([path name '.lgd'], 'Save logfile as');
				if (!ischar(files))
					logFiles = {[]};
				else
					logFiles = fullfile(path, {files});
				endif
			else
				testcaseFiles = fullfile(path, files);
				path = uigetdir(path, 'Select logfile directory');
				if (!ischar(path))
					logFiles = cell(length(files), 1);
				else
					for i = 1 : length(files)
						[~, name] = fileparts(files{i});
						logFiles{i} = fullfile(path, [name '.lgd']);
					endfor
				endif
			endif

			for i = 1 : length(testcaseFiles)
				printf('processing %s...\n', testcaseFiles{i}); fflush(stdout);
				load(testcaseFiles{i}, 'testcase');
				logData = runTests(testcase, logFiles{i});
			endfor
			printf('done\n');

		endif

	% push results
	case 6
		for r = 1 : length(logData)
			logDataStack(end+1).mixerVersion = [0,0];
			logDataStack(end).testbed = struct();
			logDataStack(end).config = logData(r).config;
			logDataStack(end).config.label = runLabels{r};
			logDataStack(end).rank = logData(r).rank;
		endfor

	% pop results
	case 7
		logData = logDataStack;
		clear logDataStack;

	% evaluate results
	case 8
		[fileName, path] = uigetfile({'*.lgd;*.dat', 'Log data files'}, 'Load logfile, cancel to reuse loaded data', 'data/');
		if (ischar(fileName))
			logFilePath = [path fileName];
			load(logFilePath, 'logData', 'testcase');
		endif

		if (exist('logData') && exist('testcase'))
			
			action = menu('choose action:', {
				'plot single round(s)',
				'plot merged rounds',
				'plot merged rounds + nodes',
				'generate activity plot',
				'plot fTxCurve',
				'export CSV file(s)'
			});

			runLabels = {};
			for run = 1 : length(logData)
				t = 1 + fix((run - 1) / length(testcase.config));
				c = 1 + rem((run - 1) , length(testcase.config));
				if (isstruct(logData(run).testbed))
					runLabels{run} = "";
				else
					[~, testbedLabel] = fileparts(logData(run).testbed);
					runLabels{run} = [testbedLabel " / "];
				endif
				if (isfield(logData(run).config, 'label') && !isempty(logData(run).config.label))
					runLabels{run} = [runLabels{run} logData(run).config.label];
				else
					runLabels{run} = [runLabels{run} sprintf("config %d", c)];
				endif
			endfor

			switch (action)

				case 1
					run = listdlg("Name", "Select runs", "ListString", runLabels, ...
						"SelectionMode", "Single", "ListSize", [600, 400]);
					if (isempty(run))
						params = [];
					else
						params = inputdlg( ...
							{	'round(s)',
								'figure control command(s)',
								'subfigure control command(s)'
							}, ...
							'plot settings', 1, ...
							{	sprintf("%d %% [%s]", logData(run).config.rounds(1), ...
									sprintf("%d, ", logData(run).config.rounds)(1:end-2)),
								'figure(); figure(gcf(), "name", sprintf("run %d single rounds", run))',
								'subplot(length(rounds_),1,%d)'
							} ...
						);
					endif
					if (length(params) > 0)

						rounds_ = eval(params{1});
						roundsIndex = lookup(logData(run).config.rounds, rounds_);
						
						eval(params{2});

						i = 1;
						for r = 1 : length(rounds_)
							eval(sprintf([params{3} ';'], i++));
							plotRankVsNode(double(logData(run).rank(:,:,roundsIndex(r))));
							title(sprintf("%s : round %d", runLabels{run}, rounds_(r)), 'interpreter', 'none');
						endfor

					endif

				case 2
					runs = listdlg("Name", "Select runs", "ListString", runLabels, ...
						"InitialValue", [1 : length(runLabels)], "ListSize", [600, 400]);
					if (isempty(runs))
						params = [];
					else
						params = inputdlg( ...
							{	'(sub)figure control command(s)'
							}, ...
							'plot settings', 1, ...
							{	'figure(run, "name", sprintf("run %%d", run)); subplot(3,1,%d)'
							} ...
						);
					endif
					if (length(params) > 0)

						for run = runs

							rank = double(logData(run).rank);

							numNodes = size(rank, 2);
							numSlots = size(rank, 1);

							rankMin = min(rank, [], 3);
							rankMax = max(rank, [], 3);
							rankMean = mean(rank, 3);

							i = 1;

							eval(sprintf([params{1} ';'], i++));
							plotRankVsNode(rankMin);
							title(sprintf("%s : min. rank merged over %d rounds", runLabels{run}, size(rank, 3)), 'interpreter', 'none');

							eval(sprintf([params{1} ';'], i++));
							plotRankVsNode(rankMax);
							title(sprintf("%s : max. rank merged over %d rounds", runLabels{run}, size(rank, 3)), 'interpreter', 'none');

							eval(sprintf([params{1} ';'], i++));
							plotRankVsNode(rankMean);
							title(sprintf("%s : mean rank merged over %d rounds", runLabels{run}, size(rank, 3)), 'interpreter', 'none');

						endfor

					endif

				case 3
					runs = listdlg("Name", "Select runs", "ListString", runLabels, ...
						"InitialValue", [1 : length(runLabels)], "ListSize", [600, 400]);
					if (isempty(runs))
						params = [];
					else
						params = inputdlg( ...
							{	'figure control command(s)',
								'subfigure control command(s)',
								'title'
							}, ...
							'plot settings', 1, ...
							{	'figure(); clf(); set(gcf(), "name", "comparison")',
								'subplot(3,1,%d)'
								''
							} ...
						);
					endif
					if (length(params) > 0)

						rankMin = zeros(rows(logData(1).rank), length(runs));
						rankMax = zeros(size(rankMin));
						rankMean = zeros(size(rankMin));

						numNodes = 0;

						for r = 1 : length(runs)

							run = runs(r);
							rank = double(logData(run).rank);
							numNodes = max(numNodes, columns(rank));

							rankMin(1:rows(rank),r) = min(min(rank, [], 3), [], 2);
							rankMax(1:rows(rank),r) = max(max(rank, [], 3), [], 2);
							rankMean(1:rows(rank),r) = mean(mean(rank, 3), 2);
							
						endfor
						
						[X, Y] = ndgrid([1:rows(rankMin)], [1:columns(rankMin)]);
						
						eval(params{1});
%						msgbox("Figure maximized?", "Check");

						annotation("textbox", [0, 0.97, 1, 0.02], "string", params{3}, ...
							"horizontalalignment", "center", "fitboxtotext", "off", "linestyle", "none", ...
							"fontweight", "bold", "fontsize", 12);
		
						format1 = {'-'; '--'; ':'; '-.'};
						format2 = [' ', '+', 'o', '*', 'x', 's', 'd', '^', 'v', '>', '<', 'p', 'h'];	% '.'
						format3 = ['r', 'g', 'b', 'k', 'm', 'y', 'c'];	% 'w'
						formatF = @(i) [
							format1{1 + rem(fix((i - 1) / (length(format2) + length(format3))), length(format1))}
							format2(1 + rem(fix((i - 1) / length(format3)), length(format2)))
							format3(1 + rem(i - 1, length(format3)))
						]';
						fmt = cellfun(formatF, num2cell(1 : length(runs)), "UniformOutput", false);
						
						eval(sprintf([params{2} ';'], 1));
						plot([1:rows(rankMin)], rankMin, fmt);
						ylabel('rank');
%						plot3(X, Y, rankMin);
%						view(0,0);
%						zlim([0, numNodes]);
%						zlabel('rank');
						xlabel('slot');
						h = legend(runLabels(runs), 'location', 'southeast');
						set(h, 'interpreter', 'none');
						grid on;
						title("min. rank merged over all nodes and rounds");

						eval(sprintf([params{2} ';'], 2));
						plot([1:rows(rankMax)], rankMax, fmt);
						ylabel('rank');
%						plot3(X, Y, rankMax);
%						view(0,0);
%						zlim([0, numNodes]);
%						zlabel('rank');
						xlabel('slot');
						h = legend(runLabels(runs), 'location', 'southeast');
						set(h, 'interpreter', 'none');
						grid on;
						title("max. rank merged over all nodes and rounds");

						eval(sprintf([params{2} ';'], 3));
						plot([1:rows(rankMean)], rankMean, fmt);
						ylabel('rank');
%						plot3(X, Y, rankMean);
%						view(0,0);
%						zlim([0, numNodes]);
%						zlabel('rank');
						xlabel('slot');
						h = legend(runLabels(runs), 'location', 'southeast');
						set(h, 'interpreter', 'none');
						grid on;
						title("mean rank merged over all nodes and rounds");

					endif

				case 4
				do
					[fileName, path] = uiputfile('data/*.slk', 'Save SYLK file as');
					if (!ischar(fileName))
						break;
					endif

					params = inputdlg( ...
						{	'run',
							'round',
							'log level (flags: 1 = incl. packets, 2 = incl. node states)',
							'SYLK file'
						}, ...
						'export settings', 1, ...
						{	'1',
							'1',
							['0b' dec2bin(testcase.logLevel)],
							[path fileName]
						} ...
					);
					if (length(params) == 0)
						break;
					endif

					run = eval(params{1});
					round_ = eval(params{2});
					logLevel = eval(params{3});
					slkFile = params{4};

					roundIndex = lookup(logData(run).config.rounds, round_);
					
					logDetails = [];
					logDetailsName = 'details';
					if (isfield(logData, 'logDetailsName'))
						logDetailsName = logData(run).logDetailsName{roundIndex};
					endif

					switch (testcase.logDetailsMode)
						case 'logData'
							logDetails = logData(end).(logDetailsName);
						case 'logFile'
							printf("loading %s from %s...\n", logDetailsName, logFilePath); fflush(stdout);
							temp = load(logFilePath, logDetailsName);
							logDetails = temp.(logDetailsName);
							temp = [];
						case 'extraFiles'
							[dir, name, ext] = fileparts(logFilePath);
							file = fullfile(dir, [name '_' logDetailsName ext]);
							printf("loading %s from %s...\n", logDetailsName, file); fflush(stdout);
							temp = load(file, [logDetailsName '_timeStamp'], logDetailsName);
							ts = temp.([logDetailsName '_timeStamp']);
							if (ts >= logData(run).startTime && ts <= logData(run).stopTime)
								logDetails = temp.(logDetailsName);
							else
								printf("timestamp mismatch:\n    %s <= %s <= %s\n", ...
									ctime(logData(run).startTime), ctime(ts), ctime(logData(run).stopTime));
							endif
							temp = [];
					endswitch

					if (isempty(logDetails) && logLevel != 0)
						printf("recomputing %s...\n", logDetailsName); fflush(stdout);						
						if (mixerVersion(1) != logData(run).mixerVersion(1) || mixerVersion(2) < logData(run).mixerVersion(2))
							error("mixerVersion not supported (requested: %s, current: %s)\n", ...
								sprintf("%d%s", logData(run).mixerVersion(1), sprintf(".%d", logData(run).mixerVersion(2:end))), ...
								sprintf("%d%s", mixerVersion(1), sprintf(".%d", mixerVersion(2:end))) ...
							);
						else
							mixerVersion = logData(run).mixerVersion;
						endif
						if (isstruct(logData(run).testbed))
							testbed = logData(run).testbed;
						else
							load(logData(run).testbed, 'testbed');
							Pnoise = 10 ^ (testbed.linkModel.Pnoise_dBm / 10);
						endif
						nodes = struct('id', num2cell([1 : rows(testbed.nodes)]));
						config = logData(run).config;
						rand('state', logData(run).RNGSeed(:,roundIndex));
						if (isfield(testcase, "simulationMode") && strcmp(testcase.simulationMode, "replayTrace"))
							[logRound, logDetails] = replayTestRound(testbed, round_, logLevel);
						else
							initiator = config.fInitiator(round_, rows(testbed.nodes));
							[logRound, logDetails] = runTestRound(testbed, initiator, logLevel);
						endif
					else
						logRound = struct();
						logRound.rank			= logData(run).rank(:,:,roundIndex);
						logRound.packetSource	= logData(run).packetSource(:,:,roundIndex);
						if (bitget(logLevel, 1))
							logRound.packetCoeffs = logData(run).packetCoeffs(:,:,:,roundIndex);
							if (logData(run).config.request.mode)
								logRound.packetRequest = logData(run).packetRequest(:,:,:,roundIndex);
							endif
						endif
					endif

					numSlots = rows(logRound.rank);
					numNodes = columns(logRound.rank);

					idx = zeros(numSlots, numNodes);
					roles = repmat(' ', numSlots, numNodes);
					for s = 1 : numSlots
						rxNodes = find(int8(logRound.packetSource(s,:)) - [1 : numNodes]);
						txNodes = setdiff([1 : numNodes], rxNodes);
						idx(s,txNodes) = [1 : length(txNodes)];
						idx(s,rxNodes) = [idx(s,:), 0](1 + mod(logRound.packetSource(s,rxNodes) - 1, numNodes + 1));
						roles(s,txNodes) = 'T';
						roles(s,rxNodes) = 'R';
					endfor

					printf('writing %s...\n', slkFile); fflush(stdout);

					[f,msg] = fopen(slkFile, 'wt');
					if (f < 0)
						error(msg);
						break;
					endif

					%https://en.wikipedia.org/wiki/SYmbolic_LinK_(SYLK)
					%http://www.exceltactics.com/definitive-guide-custom-number-formats-excel/

					fwrite(f, "ID;P\n");
					fwrite(f, "P;PGeneral\n");
					fwrite(f, "P;P[Black]@\n");
					fwrite(f, "P;P[Blue]@\n");
					fwrite(f, "P;P[Red]@\n");
					fwrite(f, "P;P[Magenta]@\n");
					fwrite(f, "P;P[Green]@\n");
					fwrite(f, "P;P[Cyan]@\n");
					fwrite(f, "P;P[Yellow]@\n");
					fwrite(f, "F;G\n");

					if (logData(run).config.fieldSize > 16)
						coeffFormat = "%02X-";
					else
						coeffFormat = "%X";
					endif
					
					y = 1;
					dx = 3;

					% slotNumber
					x = 2;
					for s = 1 : numSlots
						fprintf(f, "C;Y%d;X%d;K\"%d / %04x\"\n", y, x, s, s);
						x += dx;
					endfor		
					y += 2;

					% packets
					x = 1;
					for n = 1 : numNodes
						for s = 1 : numSlots

							fprintf(f, "C;Y%d;X%d;K\"%s:\"\nF;FD0R\n", y, x+0, roles(s,n));
							if (n == 1)
								fprintf(f, "F;W%d %d %d\n", x, x, 4);
							endif

							if (isfield(logRound, "packetCoeffs"))
								c = logRound.packetCoeffs(n,:,s);
								if (logData(run).config.request.mode)
									r = logRound.packetRequest(n,:,s);
								endif
								if (logRound.packetSource(s,n))
									fprintf(f, "C;Y%d;X%d;K\"%s\"\nF;P%d;STBLR\n", y, x+1, sprintf(coeffFormat, c), 1 + rem(idx(s,n) - 1, 7));
									if (roles(s,n) == 'R')
										fprintf(f, "F;SD\n");
									endif
									if (logData(run).config.request.mode)
										fprintf(f, "C;Y%d;X%d;K\"%s\"\nF;P%d;STBLR\n", y, x+2, sprintf("%d", r), 1 + rem(idx(s,n) - 1, 7));
										if (roles(s,n) == 'R')
											fprintf(f, "F;SD\n");
										endif
									endif
								endif
							endif
							if (n == 1)
								fprintf(f, "F;W%d %d %d\n", x+1, x+1, columns(c) * length(sprintf(coeffFormat, 0)));
								fprintf(f, "F;W%d %d %d\n", x+2, x+2, columns(c));
							endif

							x += dx;

						endfor
						y += 1; x = 1;
					endfor

					% history
					if (isfield(logDetails, 'nodes'))
						y += 1;
						for n = 1 : numNodes
							x = 2;
							for s = 1 : numSlots
								h = logDetails(s).nodes(n).history;
								h(find(uint16(h) - 9)) = '#' - '0';
								h = char(h + '0');
								fprintf(f, "C;Y%d;X%d;K\"%s\"\n", y, x, h);
								x += dx;
							endfor		
							y += 1;
						endfor
					endif

					% request marks
					if (isfield(logDetails, 'nodes'))
						y -= numNodes;
						for n = 1 : numNodes
							x = 3;
							for s = 1 : numSlots
								m  = 2 * (1 - logDetails(s).nodes(n).requestOrMask);
								m += 1 * (1 - logDetails(s).nodes(n).requestAndMask);
								m += 8 * (1 - logDetails(s).nodes(n).requestOrMask2);
								m += 4 * (1 - logDetails(s).nodes(n).requestAndMask2);
%								m = char(m + '0');
								m = sprintf("%X", m);
								fprintf(f, "C;Y%d;X%d;K\"%s\"\n", y, x, m);
								x += dx;
							endfor		
							y += 1;
						endfor
					endif

					% rank
					y += 1;
					for n = 1 : numNodes
						x = 2;
						for s = 1 : numSlots
							fprintf(f, "C;Y%d;X%d;K%d\n", y, x, logRound.rank(s,n));
							if (s > 1 && logRound.rank(s-1,n) != logRound.rank(s,n))
								fprintf(f, "F;SD\n");
							endif	
							x += dx;
						endfor		
						y += 1;
					endfor

					% matrices
					if (isfield(logDetails, 'nodes'))
						y += 1;
						for n = 1 : numNodes
							x = 1;
							for s = 1 : numSlots
								yd = 0;
								c = logDetails(s).nodes(n).coeffs;
								for k = 1 : rows(c)
									fprintf(f, "C;Y%d;X%d;K\"%s\"\n", y+yd, x+1, sprintf(coeffFormat, c(k,:)));
									if (s > 1)
										if (!isequal(c(k,:), logDetails(s-1).nodes(n).coeffs(k,:)))
											fprintf(f, "F;SD\n");
										endif
									endif	
									fprintf(f, "C;Y%d;X%d;K%d\n", y+yd, x+0, logDetails(s).nodes(n).birthSlot(k));
									if (s > 1)
										if (!isequal(c(k,:), logDetails(s-1).nodes(n).coeffs(k,:)))
											fprintf(f, "F;SD\n");
										endif
									endif	
									yd += 1;
								endfor
								x += dx;
							endfor		
							y += yd + 1;
						endfor
					endif

					fwrite(f, "E");
					fclose(f);

					system(['start ' slkFile]);

				until (true)

				case 5
					figure();
					[aa, dd] = meshgrid(0:50, 1:20);
					mesh(aa, dd, testcase.config.fTxCurve(aa, dd));
					hidden off;
					xlabel('age');
					ylabel('density');
					zlabel('tx probability');

				% export CSV file(s)
				case 6

					run = listdlg("Name", "Select run", "ListString", runLabels, ...
						"InitialValue", 1, "SelectionMode", "Single", "ListSize", [600, 400]);
					if (isempty(run))
						outFile = [];
					else
						if (exist("traceFile", "var") && !isempty(traceFile))
							[path,~] = fileparts(traceFile);
						else
							path = ".";
						endif
						[file, path] = uiputfile([path '/rank.csv'], 'Save CSV file as');
						if (!ischar(file))
							outFile = [];
						else
							outFile = fullfile(path, file);
						endif
					endif

					if (length(outFile) > 0)

						rank = double(logData(run).rank);

						numNodes = size(rank, 2);
						numSlots = size(rank, 1);

						rankMin = min(rank, [], 3);
						rankMax = max(rank, [], 3);
						rankMean = mean(rank, 3);

						outFile = fopen(outFile, "wt");
						if (outFile < 0)
							error("unable to open ouput file");
						endif

						fprintf(outFile, "slot,rank_mean,rank_min,rank_max");
						fprintf(outFile, ",rank_mean_rel,rank_min_rel,rank_max_rel");
						for k = 1 : numNodes
							fprintf(outFile, ",node_%u_rank_mean", k);
						endfor
						for k = 1 : numNodes
							fprintf(outFile, ",node_%u_rank_min", k);
						endfor
						for k = 1 : numNodes
							fprintf(outFile, ",node_%u_rank_max", k);
						endfor
						fprintf(outFile, "\n");

						csvwrite(outFile,
							[(1 : numSlots)', ...
							mean(rankMean, 2), min(rankMin, [], 2), max(rankMax, [], 2), ...
							[mean(rankMean, 2), min(rankMin, [], 2), max(rankMax, [], 2)] / numNodes, ...
							rankMean, rankMin, rankMax]
						);

						fclose(outFile);

					endif

			endswitch

		endif

	% replay trace file
	case 4

		[fileName, path] = uigetfile({'*.log;*.txt;*.csv'}, 'Load trace file');
		if (!fileName)
			return;
		endif

		traceFile = [path fileName];
		[fLineHeader, testbed, setup, rounds, traceMap] = scanTraceFile(traceFile);
		
		params = inputdlg( ...
			{	'rounds',
				'numSlots',
				'payloadSize',
				'log level (flags: 1 = incl. packets, 2 = incl. node states)',
				'save details in (0 = drop, 1 = log structure, 2 = log file, 3 = extra files)'
			}, ...
			'replay testcase parameters', 1, ...
			{	sprintf("%d:%d", rounds(1), rounds(end)),
				sprintf("%d", setup.slots),
				sprintf("%d", setup.payload),
				'0b11',
				'0'
			} ...
		);
		if (!length(params))
			return;
		endif

		testcase = struct("simulationMode", "replayTrace");
		testcase.testbeds = testbed;

		configSource = [ ...
			"config.label = \"import from " fileName "\";								\n" ...
			"config.fieldSize = 2;														\n" ...
			"config.payloadDistribution = '[1 : length(nodes)]';						\n" ...
%			"config.fTimeout = @(n) 1;	% unused in replay mode							\n" ...
			"config.historyWindowLength = 'round([3, 1] * length(nodes))';				\n" ...
			"config.exchangeTriggerSparsity = 'length(nodes)';							\n" ...
			"config.immediateElimination = false;										\n" ...
%			"config.fastTxUpdate.allowedUpdates = 0;	% unused in replay mode			\n" ...
%			"config.smartShutdown = false;	% unused in replay mode						\n" ...
			"config.recursiveNeighborhood = 0;											\n"	...
%			"config.txPacket.includeOwn = 0;	% unused in replay mode					\n" ...
%			"config.txPacket.emptyPacketStrategy = 'own';	% unused in replay mode		\n" ...
			"config.request.mode = 'column,pivot';										\n" ...
			"config.request.columnSearchMode = 'pivot';									\n" ...
			"\t% 'pivot' or 'all'														\n" ...
			"config.request.rxSnoop = true;												\n" ...
		];
		configSource = strjoin(strtrim(strsplit(configSource, "\n")), "\n");
		
		configFile = [tempname("", "oct-testcase-config-") ".m"];
		f = fopen(configFile, "w");
		fputs(f, configSource);
		fclose(f);

    system([program_invocation_name " --gui --persist --eval \"edit '" configFile "'\""], false, "sync");
		%system(configFile, false, "sync");
		b = questdlg("Ready to continue?", "Edit config options", "Cancel", "Yes", "Yes");
		if (strcmp(b, "Yes"))

			configSource = fileread(configFile);
			configSource = strrep(strrep(configSource, "\r\n", "\n"), "\r", "\n");
			configSource = strrep(strrep(configSource, "@", "@@"), "\n", "\n@");
			configSource = strjoin(strtrim(strsplit(configSource, "\n", "collapsedelimiters", false)), "\n");
			testcase.configSource = strrep(configSource, "\n@", "@n");
			% "\n@" is temporarily used to avoid that strtrim() trims something on the left (like tabs)

			configTemp = config;
%			clear config;
			config = struct();
			source(configFile);
			testcase.config = config;
			config = configTemp;
			clear("configTemp");

			% unused in replay mode
			testcase.config.fTimeout = @(n) 1;
			testcase.config.fastTxUpdate.allowedUpdates = 0;
			testcase.config.smartShutdown = false;
			testcase.config.txPacket.includeOwn = 0;
			testcase.config.txPacket.emptyPacketStrategy = 'own';

			testcase.config.traceFile	= traceFile;
			testcase.config.fLineHeader	= fLineHeader;
			testcase.config.traceMap	= traceMap;
			testcase.config.rounds 		= eval(params{1});
			testcase.config.numRounds 	= length(testcase.config.rounds);
			testcase.config.numSlots 	= eval(params{2});
			testcase.config.payloadSize = eval(params{3});

			testcase.logLevel = uint32(eval(params{4}));
			testcase.logDetailsMode = {'none', 'logData', 'logFile', 'extraFiles'}{eval(params{5}) + 1};

			delete(configFile);

			[~, name] = fileparts(traceFile);
			[file, path] = uiputfile([name '_log.dat'], 'Save logfile as');
			if (!ischar(file))
				logFile = [];
			else
				logFile = fullfile(path, file);
			endif

			printf('processing %s...\n', traceFile); fflush(stdout);
			logData = runTests(testcase, logFile);

			printf('done\n');

		else
			delete(configFile);
		endif

	% import log file
	case 5

		[fileName, path] = uigetfile({'*.log;*.txt;*.csv'}, 'Load testbed logfile');
		if (!fileName)
			return;
		endif

		traceFile = [path fileName];
		[fLineHeader, testbed, setup, rounds, traceMap] = scanTraceFile(traceFile);
		
		[~, name] = fileparts(traceFile);
		[file, path] = uiputfile([path name '_log.dat'], 'Save logfile as');
		if (!ischar(file))
			logFile = [];
		else
			logFile = fullfile(path, file);
		endif

		params = inputdlg( ...
			{	'run label',
				'rounds',
				'drop damaged rounds'
			}, ...
			'import parameters', 1, ...
			{	["import from " fileName],
				sprintf("%d:%d %% [%s]", rounds(1), rounds(end), sprintf("%d, ", rounds)(1:end-2)),
				'1'
			} ...
		);
		if (!length(params))
			return;
		endif

		rounds = eval(params{2});

		logData = [];
		logData.rank = uint8(zeros(setup.slots, length(testbed.nodes), length(rounds)));

		rank_ = logData.rank(:,:,1);

		f = fopen(traceFile, "rb");

		roundIndex = 1;
		while (roundIndex <= length(rounds))
			round_ = rounds(roundIndex);
			
			% extract trace of round <round_>
			[~, I] = sort(traceMap(:, round_));
			dataMap = traceMap(I, round_ : round_ + 1);
			fseek(f, 0, SEEK_SET);
			dataString = "";
			for i = 1 : length(testbed.nodes)
				if (dataMap(i,1) >= ftell(f))
					fseek(f, dataMap(i,1), SEEK_SET);
					dataString = [dataString, fread(f, [1, dataMap(i,2) - dataMap(i,1)], "*char")];
				elseif (dataMap(i,2) > ftell(f))
					dataString = [dataString, fread(f, [1, dataMap(i,2) - ftell(f)], "*char")];
				endif
				dataMap(i,2) -= dataMap(i,1);
				dataMap(i,1) = 1 + length(dataString) - (ftell(f) - dataMap(i,1));
				dataMap(i,2) += dataMap(i,1) - 1;
			endfor

			% read rank history from trace
			rank_(:) = 0;
			ok = true;
			pattern = sprintf("DEBUG \\(%d\\): rank up slot (\\[\\d+(?:, \\d+)*,?\\s*\\])", round_);
			for l = 1 : length(testbed.nodes)
				k = I(l);
				s = regexp(dataString(dataMap(l,1) : dataMap(l,2)), ...
						[fLineHeader(sprintf("%d", testbed.nodes(k,2))) pattern], "tokens", "once");
				if (isempty(s))
					warning("missing rank trace round %d node %d\n", round_, testbed.nodes(k,2));
					ok = false;
				else
					s = eval(s{1});
					rank_(1,k) += length(find(s == 0));
					s(find(s == 0)) = [];	
					if (find(s > rows(rank_)))
						s_ = find(s > rows(rank_));
						warning("invalid rank values round %d node %d:%s\n", ...
							round_, testbed.nodes(k,2), sprintf(" %u", s(s_)));
						s(s_) = [];
					endif
					% NOTE: s can have identical entries due to weak zeros handling
					for s_ = unique(s)
						rank_(s_,k) += length(find(s == s_));
					endfor
				endif
			endfor

			if (!ok && eval(params{3}))
				rounds(roundIndex) = [];
			else
				logData.rank(:,:,roundIndex) = cumsum(rank_, 1);
				roundIndex += 1;
			endif

		endwhile

		fclose(f);

		config = struct();
		config.label			= params{1};
		config.numSlots			= setup.slots;
		config.rounds	 		= rounds;

		testcase = struct();
		testcase.testbeds 		= testbed;
		testcase.config			= config;
		testcase.logLevel 		= 0;
		testcase.logDetailsMode = 'none';

		logData.mixerVersion	= [0,0];
		logData.testbed 		= testbed;
		logData.config 			= config;
		logData.rank 			= logData.rank(:,:,1 : length(rounds));

		if (!isempty(logFile))
			printf('writing log data to %s...\n', logFile); fflush(stdout);
			save('-text', '-zip', logFile, 'testcase');
			save('-append', '-text', '-zip', logFile, 'logData');
		endif

endswitch

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
