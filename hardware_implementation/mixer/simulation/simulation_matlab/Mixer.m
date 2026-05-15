%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

1;

global mixerVersion;
mixerVersion = [1,4];	% attention: do init independent from declaration to ensure reinitialization on every start

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

GFarithmetic;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function node = init(hNode, initiator)

	global nodes;
	global config;
	global mixerVersion;

	if (mixerVersion(1) != 1 || mixerVersion(2) > 4)
		error("unsupported mixerVersion (requested: %d.%d, supported: 1.0 - 1.4)", ...
			mixerVersion(1), mixerVersion(2));
	endif

	if (isstruct(hNode))
		node = hNode;
	else
		node = nodes(hNode);
	endif

	numNodes = length(nodes);

	if (mixerVersion(1) == 1 && mixerVersion(2) < 3)
		config.payloadDistribution			= 1 : numNodes;
		config.txPacket.emptyPacketStrategy	= 'own';
	else
		if (!isfield(config.txPacket, "emptyPacketStrategy"))
			if (!isfield(config, "payloadDistribution"))
				config.txPacket.emptyPacketStrategy = 'own';
			else
				disp(config.txPacket.emptyPacketStrategy)	% trigger error
			endif
		endif
		if (!isfield(config, "payloadDistribution"))
			config.payloadDistribution = 1 : numNodes;
		elseif (ischar(config.payloadDistribution))
			config.payloadDistribution = eval(config.payloadDistribution);
		endif
	endif

	generationSize = length(config.payloadDistribution);

	if (ischar(config.historyWindowLength))
		config.historyWindowLength = eval(config.historyWindowLength);
	endif

	if (ischar(config.exchangeTriggerSparsity))
		config.exchangeTriggerSparsity = eval(config.exchangeTriggerSparsity);
	endif

	if (mixerVersion(1) == 1 && mixerVersion(2) < 4)
		config.immediateElimination = false;
	endif

	config.fastTxUpdate.fCOnRx 		= @() uint8(1);
	config.fastTxUpdate.fCOnRequest	= @() uint8(1);
	config.fastTxUpdate.fCOnOwn 	= @() uint8(1);
	if (!isstruct(config.fastTxUpdate))
		config.fastTxUpdate = struct("allowedUpdates", int8(config.fastTxUpdate != false));
	elseif (config.fastTxUpdate.allowedUpdates > 0)
		if (config.fastTxUpdate.mulOnRx)
			config.fastTxUpdate.fCOnRx = @() uint8(randi(config.fieldSize - 1));
		endif	
		if (config.fastTxUpdate.mulOnRequest)
			config.fastTxUpdate.fCOnRequest = @() uint8(randi(config.fieldSize - 1));
		endif	
		if (config.fastTxUpdate.mulOnOwn)
			config.fastTxUpdate.fCOnOwn = @() uint8(randi(config.fieldSize - 1));
		endif	
	endif
	
	if (!index(config.request.mode, 'c'))
		config.request.fTxColumnYesNo = @() false;
	endif
	if (!index(config.request.mode, 'p'))
		config.request.fTxPivotYesNo = @() false;
	endif

	if (isequal(mixerVersion, [1,1]))
		config.fieldSize = 2;
	endif
	[p,n] = factor(config.fieldSize);
	if ((length(p) != 1) || (n > 1 && p != 2))
		error("invalid fieldSize\np = %sn = %s", disp(p), disp(n));
	endif
	fieldSizeBits = ceil(log2(p)) * n;

	if (n > 1)
		config.GFadd = @(a,b) GF2n_add(a,b,n);
		config.GFsub = @(a,b) GF2n_sub(a,b,n);
		config.GFmul = @(a,b) GF2n_mul(a,b,n);
		config.GFdiv = @(a,b) GF2n_div(a,b,n);
		config.GFmmul = @(a,b) GF2n_mmul(a,b,n);
	elseif (p > 2)
		config.GFadd = @(a,b) GFp_add(a,b,p);
		config.GFsub = @(a,b) GFp_sub(a,b,p);
		config.GFmul = @(a,b) GFp_mul(a,b,p);
		config.GFdiv = @(a,b) GFp_div(a,b,p);
		config.GFmmul = @(a,b) GFp_mmul(a,b,p);
	else
		config.GFadd = @(a,b) GF2_add(a,b);
		config.GFsub = @(a,b) GF2_sub(a,b);
		config.GFmul = @(a,b) GF2_mul(a,b);
		config.GFdiv = @(a,b) GF2_div(a,b);
		config.GFmmul = @(a,b) GF2_mmul(a,b);
	endif

	node.coeffs		= uint8(zeros(generationSize));
	node.payloads	= uint8(zeros(generationSize, config.payloadSize * 8 / fieldSizeBits));
	node.rank		= 0;
	% notice: fieldSizeBits is an upper bound, so the payload length should be stretched
	% if p is not a power of 2 to account for the overhead of a precoding scheme. However,
	% since the real overhead is depending on the precoding scheme, we don't consider this here
	% (we assume that it is realized on a higher layer). Since the payload length is not very
	% important for the simulation results, this is no critical issue.

	for i = find(config.payloadDistribution == node.id)
		d = GFunpack(uint8(i), config.fieldSize)(end:-1:1);
		node.coeffs(i,i) = 1;
		node.payloads(i, 1 : length(d)) = d;
		node.rank += 1;
	endfor

	node.state 			= -1;
	node.birthSlot		= zeros(generationSize, 1);
	node.history		= zeros(1, numNodes);
	node.numTx			= 0;
	node.neighborhood	= uint8(ones(1, numNodes));
	node.requestOrMask	= logical(ones(1, generationSize));
	node.requestAndMask	= logical(ones(1, generationSize));
	node.requestOrMask2	= logical(ones(1, generationSize));
	node.requestAndMask2= logical(ones(1, generationSize));
	node.requestUpdateSlot	= -1;
	node.pivotMap		= logical(zeros(numNodes, generationSize));
	node.rankMap		= uint8(zeros(1, numNodes));
	node.includeOwn		= config.txPacket.includeOwn;

	node = prepareTxPacket(node, 0);

	% select initiator
%	if (node.id == config.initiator)
	if (node.id == initiator)
		if (node.rank == 0)
			error("initiator (node %d) has no data. check config.payloadDistribution", node.id);
		endif
		node.state = 0;		
	endif

	if (!isstruct(hNode) && nargout < 1)
%		nodes(hNode) = node;
		f = fieldnames(node);
		for i = 1 : length(f)
			nodes(hNode).(f{i}) = node.(f{i});
		endfor
	endif

endfunction

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function node = prepareTxPacket(hNode, slotNumber)

	global nodes;
	global config;
	global mixerVersion;

	if (isstruct(hNode))
		node = hNode;
	else
		node = nodes(hNode);
	endif

	% don't do anything in replay mode (not needed)
	% note: advantages of catching this here:
	% - associated config options are not needed
	% - don't waste time
	if (strcmp(config.simulationMode, "replayTrace"))
		if (slotNumber <= 0)
			node.txPacket = [];
			if (!isstruct(hNode) && nargout < 1)
				nodes(hNode).txPacket = node.txPacket;
			endif
		endif
		return;
	endif

	I = Ic = [];
if (slotNumber < columns(node.coeffs) && node.rank > node.numTx)
	I = find(diag(node.coeffs));
	[~,I2] = sort(node.birthSlot(I));
	I = I(I2(1 : node.numTx + 1));
%			I = I(1 : lookup(I, node.numTx + 1));
%		I = I(1 : min(slotNumber + 1, length(I)));
%slotNumber, node.id, I
else
%endif
%if (slotNumber >= columns(node.coeffs))
%	if (slotNumber > 0)

		if (index(config.request.mode, 'p'))
			i = 0;
			if (!all(node.requestAndMask2))
				for  m = find(!node.requestOrMask2)(end:-1:1)
					if (node.coeffs(m,m) != 0)
						i = m;
						break;
					endif
				endfor
				if (!i)
					for  m = find(!node.requestAndMask2)(end:-1:1)
						if (node.coeffs(m,m) != 0)
							i = m;
							break;
						endif
					endfor
				endif
			endif
			if (i)
				I = [i];
				Ic = [1 : i];
			else
				node.requestOrMask2(:) = true;
				node.requestAndMask2(:) = true;
			endif
		endif

		for i = setdiff(1 : rows(node.coeffs), Ic)
			if (node.coeffs(i,i) != 0)
%				if (i == node.id && config.txPacket.includeOwn)
%					p = 2;
%				else
					age = slotNumber - node.birthSlot(i);
					p = config.txPacket.fAgeToP(age);
%				endif
				if (rand() <= p)
					I(end + 1) = i;
				endif	
			endif	
		endfor

	endif
	if (isempty(I) && node.rank > 0)
		switch (config.txPacket.emptyPacketStrategy)
			case "own"
				I = [node.id];
			case "first"
				I = [find(diag(node.coeffs), 1)];
			case "random"
				I = find(diag(node.coeffs));
				I = [I(randperm(length(I), 1))];
			otherwise
				error("config.txPacket.emptyPacketStrategy is invalid: %s", disp(config.txPacket.emptyPacketStrategy));
		endswitch
	endif

	% note: don't update requestMask2(I(1)) since
	% - the flag might be used to trigger transmission
	% - the packet may be updated before transmission by fastUpdate processing
	% -> update all requestMasks together with transmission

	txPacket.source  = uint8(node.id);
	txPacket.flags   = uint8(0);
	if (config.recursiveNeighborhood)
		txPacket.density = uint8(1);
	endif
	if (config.request.mode)
		txPacket.requestField = uint8(zeros(1, rows(node.coeffs)));
	endif

	if (isempty(I))
		txPacket.coeffs	 = uint8(zeros(1, columns(node.coeffs)));
		txPacket.payload = uint8(zeros(1, columns(node.payloads)));
	else
		if (config.fieldSize == 2)
			c = ones(1, length(I));
		else
			c = randi(config.fieldSize - 1, 1, length(I));
		endif
		txPacket.coeffs = config.GFmmul(c, node.coeffs(I,:));
		txPacket.payload = config.GFmmul(c, node.payloads(I,:));

		if (node.includeOwn && txPacket.coeffs(node.id) == 0)
			txPacket.coeffs = config.GFadd(txPacket.coeffs, node.coeffs(node.id,:));
			txPacket.payload = config.GFadd(txPacket.payload, node.payloads(node.id,:));
		endif
	endif

	% note: txPacket is not a zero packet for sure since matrix rows are linearly independent
	node.txPacket = txPacket;
	
	if (!isstruct(hNode) && nargout < 1)
		nodes(hNode).txPacket = node.txPacket;
	endif

endfunction

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function txPacket = transmit(iNode, slotNumber)

	global nodes;
	global config;
	node = nodes(iNode);

%printf("slot %d node %d state %d\n", slotNumber, node.id, node.state)

	txPacket = [];

	% switch (state)

	% not initiated or done -> Rx until something received
	if (node.state < 0)		
		;

	% timeout running -> update timeout, Rx
	% notice: we can continue with Rx here because Tx decisions are realized in Rx routine by using short timeouts
	elseif (node.state > 0)	
		nodes(iNode).state -= 1;

	% Tx
	else

		txPacket = node.txPacket;

		% call receive() for further processing (bundled there)
		receive(iNode, [], slotNumber, true);

	endif

endfunction

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function node = receive(hNode, packet, slotNumber, calledFromTx = false, nextTxPacket)

	global mixerVersion;
	global nodes;
	global config;
%if (slotNumber == 300) hNode, tic(); endif
%if (slotNumber == 300) toc(); endif
	if (isstruct(hNode))
		node = hNode;
	else
		node = nodes(hNode);
	endif
%if (slotNumber == 300) toc(); endif
	% if we are done: stop
	if (node.state < -1)
		return;
	endif

	numNodes = length(nodes);
	generationSize = rows(node.coeffs);
	allowedTxUpdates = config.fastTxUpdate.allowedUpdates;
	payloadOwnerFlag = 0;

%	% nominal timeout: 3...7 slots from now
%	nextTimeout = 2 + randi(5) - 1;
	nextTimeout = config.fTimeout(numNodes);

	% update history counters
	node.history(find(node.history)) -= 1;

	if (node.rank == generationSize)
		if (!any(node.history))
			if (find(node.rankMap == generationSize, 1))
				node.rankMap(node.id) = generationSize;
			endif
		endif
	endif

	% if we used current slot for Tx
	if (calledFromTx)

		node.numTx += 1;

		% handle smartShutdown
%TODO: auf Tx-Handling im vorhergehenden Slot verlagern damit Markierung mitgesendet werden kann,
%dann bei Rx mit Markierung Knoten sofort aus History entfernen (wichtig für Sendeentscheidung beim Helfen)
		if (config.smartShutdown)
			if (node.rankMap(node.id) == generationSize && ...
			!any(node.history .* (generationSize - node.rankMap)))
%TODO: funktioniert nicht ohne request enabled und numNodes > generationSize -> selbst-ACK einbauen			
%			if (slotNumber >= 2 * generationSize && node.rank == generationSize && ...
%			!any(node.history .* (node.neighborhood != 0)))
			% notice: checking slotNumber is very important in 1-to-all scenarios
			% since initiator would shutdown immediately after first transmission without it
			% TODO: besseres Kriterium finden
				node.state = -2;
				node.history(:) = 0;		% as marker for activity plot
				node.history(node.id) = 2;	% as marker for activity plot
				if (!isstruct(hNode) && nargout < 1)
					nodes(hNode) = node;
				endif
				return;
			endif
		endif
%if (slotNumber == 300) "tx", toc(); endif
		% use slot to prepare next Tx packet
		node = prepareTxPacket(node, slotNumber);
%if (slotNumber == 300) toc(); endif
	% if we received nothing
	elseif (isempty(packet))

		% if we are not yet initiated: don't change state
		if (node.state < 0)
			return;
		endif

		% else keep current timeout running (if Tx decision is 'no' for next slot)
		nextTimeout = node.state;

	% if we received something: process it
	else
%if (slotNumber == 300) "rx", toc(); endif
%		if (node.rank == generationSize || bitand(packet.flags, 0x80))
		if (bitand(packet.flags, 0x80))
			node.history(packet.source) = config.historyWindowLength(end);
		else
			node.history(packet.source) = config.historyWindowLength(1);
		endif

		if (bitand(packet.flags, 0x80))
			node.neighborhood(packet.source) = 0;
node.pivotMap(packet.source,:) = true;
node.rankMap(packet.source) = generationSize;
		elseif (config.recursiveNeighborhood)
			node.neighborhood(packet.source) = max(1, packet.density);
		endif

		payloadOwnerFlag = bitand(packet.flags, 0x40);
		
		% notice: includeOwn can also be negative
		if (node.includeOwn > 0 && packet.coeffs(node.id) != 0)
			node.includeOwn -= 1;
		endif

		% if not full rank: process received data
		if (node.rank < generationSize)

			rxPacket = packet;

			% extend matrix
			while (1)

				i = find(packet.coeffs, 1);
				if (isempty(i))
					break;
				endif
			
				if (node.coeffs(i,i) == 0)	

					node.coeffs(i,:) = packet.coeffs;
					node.payloads(i,:) = packet.payload;
					node.rank += 1;

					node.birthSlot(i) = slotNumber;

					if (allowedTxUpdates > 0)
						c = config.fastTxUpdate.fCOnRx();
						node.txPacket.coeffs = config.GFadd(node.txPacket.coeffs, config.GFmul(c, node.coeffs(i,:)));
						node.txPacket.payload = config.GFadd(node.txPacket.payload, config.GFmul(c, node.payloads(i,:)));
						allowedTxUpdates -= 1;
					endif

					break;
				endif

				c = nnz(packet.coeffs);
				if (c < nnz(node.coeffs(i,:)) && c <= config.exchangeTriggerSparsity)
					t = packet;
					packet.coeffs 		= node.coeffs(i,:);
					packet.payload 		= node.payloads(i,:);
					node.coeffs(i,:) 	= t.coeffs;
					node.payloads(i,:)	= t.payload;
				endif

				c = config.GFdiv(packet.coeffs(i), node.coeffs(i,i));
%				packet.coeffs(i:end) = config.GFsub(packet.coeffs(i:end), config.GFmul(c, node.coeffs(i,i:end)));
				packet.coeffs = config.GFsub(packet.coeffs, config.GFmul(c, node.coeffs(i,:)));
				packet.payload = config.GFsub(packet.payload, config.GFmul(c, node.payloads(i,:)));
				% attention: when working with real values it is important to remove accuracy errors here
				% (i.e. set packet.coeffs(i) = 0 explicitly after computations)

			endwhile

			% use received packet for immediate elimination
			if (i && config.immediateElimination)

				% make pivot element = 1
				if (node.coeffs(i,i) != 1)
					c = config.GFdiv(1, node.coeffs(i,i));
					node.coeffs(i,:) = config.GFmul(c, node.coeffs(i,:));
					node.payloads(i,:) = config.GFmul(c, node.payloads(i,:));
				endif

				% substitute downwards
				for k = i + find(diag(node.coeffs)(i + 1 : end))'
					c = node.coeffs(i,k);
					if (c != 0)
						node.coeffs(i,:) = config.GFsub(node.coeffs(i,:), config.GFmul(c, node.coeffs(k,:)));
						node.payloads(i,:) = config.GFsub(node.payloads(i,:), config.GFmul(c, node.payloads(k,:)));
					endif
				endfor

				% eliminate upwards
				for k = find(node.coeffs(1 : i - 1, i))'
					c = node.coeffs(k,i);
					node.coeffs(k,:) = config.GFsub(node.coeffs(k,:), config.GFmul(c, node.coeffs(i,:)));
					node.payloads(k,:) = config.GFsub(node.payloads(k,:), config.GFmul(c, node.payloads(i,:)));
				endfor

			endif

			% if full rank reached: solve (decode)
			if (node.rank == generationSize)
			
				for i = generationSize : -1 : 1

					c = config.GFdiv(1, node.coeffs(i,i));
					node.coeffs(i,:) = config.GFmul(c, node.coeffs(i,:));
					node.payloads(i,:) = config.GFmul(c, node.payloads(i,:));

					for k = i + 1 : length(packet.coeffs)
						c = node.coeffs(i,k);
						if (c != 0)
							node.coeffs(i,:) = config.GFsub(node.coeffs(i,:), config.GFmul(c, node.coeffs(k,:)));
							node.payloads(i,:) = config.GFsub(node.payloads(i,:), config.GFmul(c, node.payloads(k,:)));
						endif
					endfor
					
				endfor

%				if (length(config.historyWindowLength) > 1)
%					node.history = min(node.history, config.historyWindowLength(end));
%				endif

				% if there are no known unfinished nodes: 
				% tx in next slot to tell our potential helper that we are done;
				% use own history entry as marker (see tx decision below)
				if (config.smartShutdown)
					if (!any(node.history .* (node.neighborhood != 0)))
						node.history(node.id) = 1;
					endif
				endif

			endif

% TODO: später aktualisieren und verwenden, um Fertigstellung im aktuellen Slot zu markieren (für smartShutdown u.ä.)
if (node.rank != generationSize)
			node.rankMap(node.id) = node.rank;
endif			
			packet = rxPacket;

		endif

		% process request field
		if (config.request.mode)

			if (config.request.rxSnoop)
t1 = nnz([node.requestAndMask; node.requestAndMask2]);
				node.requestOrMask |= packet.coeffs;
				node.requestAndMask |= packet.coeffs;
				i = find(packet.coeffs, 1);
				node.requestOrMask2(i) = true;
				node.requestAndMask2(i) = true;
t2 = nnz([node.requestAndMask; node.requestAndMask2]);
if (t1 != t2)
	node.requestUpdateSlot = slotNumber;
endif
%if (t1 == t2)
%				node.requestOrMask(:) = true;
%				node.requestAndMask(:) = true;
%				node.requestOrMask2(:) = true;
%				node.requestAndMask2(:) = true;
%endif	
			endif

			mode = bitand(packet.flags, 3);
			switch (mode)
				% no mask
				case 0
					% do nothing
if (!bitand(packet.flags, 0x80))					
	node.pivotMap(packet.source, :) = packet.requestField;
	node.rankMap(packet.source) = nnz(packet.requestField);
elseif (numNodes <= generationSize)
	I = find(packet.requestField(1:numNodes));
	node.pivotMap(I, :) = true;
	node.rankMap(I) = generationSize;
endif
				% column mask
				case 1
					if (all(node.requestAndMask))
						node.requestOrMask(:) = false;
					endif
					node.requestOrMask |= packet.requestField;
					node.requestAndMask &= packet.requestField;
node.requestUpdateSlot = slotNumber;
				% pivot mask
				case 2
					if (all(node.requestAndMask2))
						node.requestOrMask2(:) = false;
					endif
					node.requestOrMask2 |= packet.requestField;
					node.requestAndMask2 &= packet.requestField;
node.pivotMap(packet.source, :) = packet.requestField;
node.rankMap(packet.source) = nnz(packet.requestField);
node.requestUpdateSlot = slotNumber;
				otherwise
					%error("invalid flag field\n");
          disp("invalid flag field\n")
			endswitch
		endif

	endif
%if (slotNumber == 300) toc(); endif
	% in replay mode: take over next packet right before tx
	% note: it is important to do this here (not below)
	% because the selection of i potentially evaluates it
	if (nargin >= 5 && !isempty(nextTxPacket))
		node.txPacket = nextTxPacket;
	endif

if (slotNumber - node.requestUpdateSlot > 2)
	node.requestOrMask(:) = true;
	node.requestAndMask(:) = true;
	node.requestOrMask2(:) = true;
	node.requestAndMask2(:) = true;
endif

	i = -(generationSize + 1);
if (slotNumber >= generationSize)	
	if (index(config.request.mode, 'c'))
		if (!all(node.requestAndMask))
			i = 0;
			switch (config.request.columnSearchMode)
				case 'pivot'
					for m = find(!node.requestOrMask)
						if (node.coeffs(m,m) != 0)
							i = m;
							break;
						endif
					endfor
					if (!i)
						for m = find(!node.requestAndMask)
							if (node.coeffs(m,m) != 0)
								i = m;
								break;
							endif
						endfor
					endif
				case 'all'
					mask = uint8(max(node.coeffs) != 0);
					m = find(!node.requestOrMask & mask, 1);
					for k = m : -1 : 1
						if (node.coeffs(k,m) != 0)
							i = k;
							break;
						endif
					endfor	
					if (!i)
						m = find(!node.requestAndMask & mask, 1);
						for k = m : -1 : 1
							if (node.coeffs(k,m) != 0)
								i = k;
								break;
							endif
						endfor	
					endif
				otherwise
					error("invalid columnSearchMode\n");
			endswitch
			iColumn = m;
		endif
	endif
	if (i <= 0 && index(config.request.mode, 'p'))
		if (!all(node.requestAndMask2))
			i = 0;
			for m = find(!node.requestOrMask2)
				if (node.coeffs(m,m) != 0)
					i = -m;
					break;
				endif
			endfor
			if (!i)
				for m = find(!node.requestAndMask2)
					if (node.coeffs(m,m) != 0)
						i = -m;
						break;
					endif
				endfor
			endif
%neighbors = find(node.history);
%m = sum(node.pivotMap(neighbors, :));
%[~,i2] = min(m + ((node.requestAndMask2 | !diag(node.coeffs)') .* numNodes));
%if (node.coeffs(i2,i2) != 0)
%	i = -i2;
%endif	
			if (i)
				if (!isempty(node.txPacket))
					k = find(node.txPacket.coeffs, 1);
					if (isempty(k))
						warning("zero packet detected, sent by %d. node in slot %d\n", node.id, slotNumber + 1);
					elseif (!(	...
						-i == k ||	...
						!calledFromTx && (isempty(packet) || node.rank == generationSize) || ...
						-i < k && allowedTxUpdates > 0	...
						))
						i = 0;
					endif
				endif
			endif
		endif
	endif
endif	
%if (slotNumber == 300) toc(); endif
	% decide what to do in next slot
	if (nargin >= 5)
		% in replay mode
		node.state = isempty(nextTxPacket);
	else

		age = slotNumber - max(node.birthSlot);
		d = 1 + max(1, nnz(node.history));

		if (abs(i) > generationSize)
			h = 0;
		else
			h = node.rank / generationSize;
			if (i == 0)
				h *= -1;
			else
%				numFullRankNeighbors = nnz(node.neighborhood(find(node.history)) == 0);
%				h /= 1 + numFullRankNeighbors;
%				h /= 1 + nnz(node.pivotMap(find(node.history), abs(i)));		% TODO: iColumn
				h /= max(1, 1 + nnz(node.pivotMap(find(node.history), abs(i))) - 1);	% TODO: iColumn
			endif
		endif

		if (calledFromTx)
			p = 0;
%		elseif (i == 0)
%			p = config.request.fpHelpless(age, d, numNodes);
%		else
%			p = config.fTxCurve(age, d, numNodes, nonzeros(node.neighborhood)');
%		endif
		else
			p = config.fTxCurve(age, d, numNodes, nonzeros(node.neighborhood)', h);
		endif

		if (config.coordinatedSlotting.on)

			% determine owner
			n = 1 + rem(slotNumber - 1 + 1, numNodes);

if (slotNumber < generationSize)
	if (n != node.id && !calledFromTx)
		if (payloadOwnerFlag)
%			p = config.fTxCurve(age, d - 1, numNodes, nonzeros(node.neighborhood)', h);
			n = node.id;
		else
			n = slotNumber + 1;
			if (node.coeffs(n,n) != 0 && node.birthSlot(n) == 0)
				n = node.id;
			else
				p = 1 / n;
				n = 0;
			endif
		endif
	endif
%if (slotNumber == 1 && node.id == 3), packet, payloadOwnerFlag, d, h, n, p, stopp, endif
%	n = 0;
%	if (!calledFromTx)
%		if (payloadOwnerFlag)
%%			p = config.fTxCurve(age, d - 1, numNodes, nonzeros(node.neighborhood)', h);
%			n = node.id;
%		else
%			n = slotNumber + 1;
%			p = 1 / n;
%			if (node.coeffs(n,n) != 0 && node.birthSlot(n) == 0)
%				n = node.id;
%			else
%				n = 0;
%			endif
%		endif
%		% TODO: während Init zusätzlich denkbar:
%		% - jeder Sender markiert, ob ihm nächster Payload gehört
%		% - wenn dem so ist, dann weiß Empfänger, dass nächster Slot nicht vom Payload-Besitzer genutzt wird,
%		%   da jener wegen calledFromTx verriegelt ist -> Slot ist concurrent nutzbar
%		% -> beschleunigt Payload-Weiterleitung im 1:n-Fall bzw. allgemeiner dann, wenn einzelne Knoten
%		%    Blöcke aufeinanderfolgender Payloads besitzen
%	endif
%	n = 0;
%	if (!calledFromTx)
%		n = slotNumber + 1;
%		p = 1 / n;
%		if (node.coeffs(n,n) != 0 && node.birthSlot(n) == 0)
%			n = node.id;
%		endif
%	endif
%	h = 0;
endif	

			% my slot
			if (n == node.id)
%				if (slotNumber <= numNodes)	% TODO: nach oben nehmen
%					h = 0;
%				endif
%				h = (slotNumber > numNodes && i == 0) * node.rank / generationSize;
				p = config.coordinatedSlotting.fpOwn(p, d, numNodes, h);

			elseif (!calledFromTx)

				% foreign slot
				if (n > 0 && node.history(n) > 0)
%					h = (i != 0 && abs(i) <= generationSize) * node.rank / generationSize;
%numFullRankNeighbors = nnz(node.history) - nnz((node.history != 0) .* (node.neighborhood != 0));
%h /= 1 + numFullRankNeighbors;
%h *= d + 1;
					p = config.coordinatedSlotting.fpForeign(p, d, numNodes, h);
					nextTimeout = max(nextTimeout, 1);	% TODO: verhindert mögliche Kollision nach leerem Rx, könnte aber in dynamischen Netzen gefährlich sein

				% concurrent arbitration slot
				else
%					if (slotNumber < numNodes)
%						p = config.coordinatedSlotting.fpInit(p, d, numNodes);
%% TODO: bei init requestMasks auf eigene Payloads initialisieren?
%					endif
				endif

			endif
%if (node.id == 1), slotNumber, n, p, endif
%if (slotNumber < numNodes)
%	if (slotNumber - max(node.birthSlot) == 0)
%		p = 0;
%	elseif (calledFromTx && slotNumber - max(node.birthSlot) > 3)
%		p = 0;
%	elseif (node.rank > node.numTx)
%		p = 2;
%	else
%		p = 1 / numNodes;
%	endif
%
%%	if (calledFromTx)
%%		p = 0;
%%	elseif (slotNumber - max(node.birthSlot) < 2)
%%		p = 0;
%%	elseif (node.rank > node.numTx)
%%		p = 2;
%%	else
%%		p = 1 / numNodes;
%%	endif
%endif

		endif
%if (node.id == 5), slotNumber, i, h, p, node.pivotMap, endif
		% own history entry is used to force tx in special situations
		if (node.history(node.id))
			p = 2;
		endif

		if (rand() < p)
			% Tx in next slot
			node.state = 0;
		else
			% Rx in next slot
			node.state = nextTimeout;
		endif

	endif
%if (slotNumber == 300) toc(); endif
	% if we didn't do expensive computations in current slot (i.e. we have some time):
	% chance to do something usefull
	if (!calledFromTx && (isempty(packet) || node.rank == generationSize))

		% TODO: if (i > 0), we could do a fastUpdate here (instead of below in case we Tx next slot)
		% -> use available fastUpdate time in every slot (not only Tx)

		if (i < 0 && -i <= generationSize)
			node = prepareTxPacket(node, slotNumber);
%TODO: i = find(...) optional; wenn nicht, dann wird unten nochmal fastTxUpdate gemacht
		endif

	endif

	% if we transmit in next slot: finalize tx packet
	if (node.state == 0)
%if (slotNumber == 300) "tx2", toc(); endif
		% if not in replay mode (else we already took the next packet above)
		if (nargin < 5)

			if (node.rank == generationSize)
				node.txPacket.flags += uint8(0x80);
			endif

			if (config.recursiveNeighborhood)
				node.txPacket.density = uint8(d);
				% TODO: schaltbar: merge(d,dn) (vgl. fTxCurve)
			endif

if (config.coordinatedSlotting.on)	
	if (slotNumber + 1 < generationSize)
		n = slotNumber + 2;
		if (node.coeffs(n,n) != 0 && node.birthSlot(n) == 0)
			node.txPacket.flags += uint8(0x40);
		endif
	endif			
endif
			
			if (config.request.mode)

				m = uint8(max(node.coeffs) != 0);

%				doC = config.request.fTxColumnYesNo(age, node.rank, generationSize);
%				doP = config.request.fTxPivotYesNo(age, node.rank, generationSize);
if (node.rank == generationSize || slotNumber < generationSize)	% TODO: letzteres nur bei coord
	doC = doP = false;
else
	doC = doP = false;
	x = (1 - 2 ^ (-0.125 * (generationSize - node.rank))) / (1 - 2 ^ (-0.125));
	if (age >= x ||
%	if (config.request.fTxColumnYesNo(age, node.rank, generationSize) ||
		!any(node.history .* (generationSize - node.rankMap)))
		if (rand() < 2 ^ -(length(m) - nnz(m)))
			doP = true;
		else
			doC = true;
		endif
	endif
endif
%				if (doC && doP)
%					if (config.request.fTxSelect(m))
%						doC = false;
%					else
%						doP = false;
%					endif
%				endif

				if (doC)
					node.txPacket.flags += uint8(1);
					node.txPacket.requestField = m;
				elseif (doP)
					node.txPacket.flags += uint8(2);
					node.txPacket.requestField = uint8(diag(node.coeffs) != 0)';
				else
					node.txPacket.flags += uint8(0);
%					node.txPacket.requestField = zeros(1, generationSize);
if (node.rank == generationSize && numNodes <= generationSize)
	node.txPacket.requestField(1:numNodes) = (node.rankMap == generationSize);
else	
					node.txPacket.requestField = uint8(diag(node.coeffs) != 0)';
endif					
				endif

				if (abs(i) <= generationSize)
					if (i > 0)
						if (allowedTxUpdates > 0 && node.txPacket.coeffs(iColumn) == 0)
							c = config.fastTxUpdate.fCOnRequest();
							node.txPacket.coeffs = config.GFadd(node.txPacket.coeffs, config.GFmul(c, node.coeffs(i,:)));
							node.txPacket.payload = config.GFadd(node.txPacket.payload, config.GFmul(c, node.payloads(i,:)));
							allowedTxUpdates -= 1;
						endif
node.requestUpdateSlot = slotNumber;
					else
						if (mixerVersion(1:2) == [1,0])
							% we don't support mixerVersion 1.0 anymore because it was buggy
							error("mixerVersion 1.0 is deprecated\n");
							% node.requestOrMask(:) = true;
							% node.requestAndMask(:) = true;
						endif
						if (i < 0)
							i *= -1;
							if (allowedTxUpdates > 0 && i < find(node.txPacket.coeffs, 1))
								c = config.fastTxUpdate.fCOnRequest();
								node.txPacket.coeffs = config.GFadd(node.txPacket.coeffs, config.GFmul(c, node.coeffs(i,:)));
								node.txPacket.payload = config.GFadd(node.txPacket.payload, config.GFmul(c, node.payloads(i,:)));
								allowedTxUpdates -= 1;
							endif
							i *= -1;
node.requestUpdateSlot = slotNumber;
						endif
					endif
				endif
				
				if (mixerVersion(1:2) == [1,0])
					% we don't support mixerVersion 1.0 anymore because it was buggy
					error("mixerVersion 1.0 is deprecated\n");
					
					% node.requestOrMask |= node.txPacket.coeffs;
					% node.requestAndMask |= node.txPacket.coeffs;

					% i = find(node.txPacket.coeffs, 2);
					% if (i(1) != node.id)
					% 	i = i(1);
					% endif
					% node.requestOrMask2(i) = 1;
					% node.requestAndMask2(i) = 1;
				endif
				
			endif

			if (node.includeOwn)
				if (allowedTxUpdates > 0 && node.txPacket.coeffs(node.id) == 0)
					c = config.fastTxUpdate.fCOnOwn();
					node.txPacket.coeffs = config.GFadd(node.txPacket.coeffs, config.GFmul(c, node.coeffs(node.id,:)));
					node.txPacket.payload = config.GFadd(node.txPacket.payload, config.GFmul(c, node.payloads(node.id,:)));
				endif
			endif

		endif
		
		if (config.request.mode)
			if (mixerVersion(1) > 1 || mixerVersion(2) >= 1)

				if (abs(i) <= generationSize && i <= 0)
					node.requestOrMask(:) = true;
					node.requestAndMask(:) = true;
				endif

if (i > 0)
	node.requestAndMask(i) = true;
	node.requestAndMask2 &= node.requestAndMask;
	node.requestOrMask(:) = true;
	node.requestAndMask(:) = true;
else
				node.requestOrMask |= node.txPacket.coeffs;
				node.requestAndMask |= node.txPacket.coeffs;
endif

				i = find(node.txPacket.coeffs, 2);
				if (isempty(i))
					warning("zero packet detected, sent by %d. node in slot %d\n", node.id, slotNumber + 1);
				else
	%				if (i(1) != node.id)
					if (i(1) != node.id || !node.includeOwn)
						i = i(1);
					endif

					node.requestOrMask2(i) = 1;
					node.requestAndMask2(i) = 1;
				endif
				
			endif
		endif

	endif
%if (slotNumber == 300) toc(); endif
	if (!isstruct(hNode) && nargout < 1)
		nodes(hNode) = node;
	endif
%if (slotNumber == 300) toc(); endif
endfunction

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
