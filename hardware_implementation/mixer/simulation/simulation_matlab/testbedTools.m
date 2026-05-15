%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

1;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function p = d2p(linkModel, d)

	if (isna(d))
		p = d;
	else	
		gain = linkModel.d2g(d);
		SINR = 10 * log10(gain) + linkModel.Ptx_dBm - linkModel.Pnoise_dBm;
		p = linkModel.SINR2p(SINR);
	endif
		
endfunction
	
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function d = p2d(linkModel, p)

	if (isna(p))
		d = p;
	else	
		SNR = fzero(@(s) linkModel.SINR2p(s) - p, [-50, 50]);
		gain_dB = linkModel.Pnoise_dBm + SNR - linkModel.Ptx_dBm;
		gain = 10 ^ (gain_dB / 10);
		d = linkModel.g2d(gain);
	endif
	
endfunction
	
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function testbed = generateTestbed(linkModel, area, numNodes, options = struct())

	testbed = struct();
	testbed.linkModel		= linkModel;
	testbed.area 			= area;
	testbed.generateOptions	= options;

	% merge default values
	dmin = sqrt(2);
	if (isfield(options, 'dmin'))
		dmin = options.dmin;
	endif

	if (isfield(options, 'dmax') && !isstruct(options.dmax))
		dmax = options.dmax;
	else
		if (isfield(options.dmax, 'SNR'))
			dmax_SNR = options.dmax.SNR;
		else
			p = 0.9;
			if (isfield(options.dmax, 'p'))
				p = options.dmax.p;
			endif
			dmax_SNR = fzero(@(s) testbed.linkModel.SINR2p(s) - p, [-20, 50]);
		endif
		% calculate maximum distance between nodes (implied by dmax_SNR)
		linkmin_dB = linkModel.Pnoise_dBm + dmax_SNR - linkModel.Ptx_dBm;
		linkmin = 10 ^ (linkmin_dB / 10);
		dmax = linkModel.g2d(linkmin);
	endif

	if (dmax <= dmin)
		error("dmin >= dmax, impossible parameter constellation");
	endif

	testbed.dmax = dmax;

	% generate nodes
	pos = zeros(numNodes, 2);
	d = zeros(numNodes);
	pos(1,:) = [randi(area(1)), randi(area(2))];
	d(1,1) = Inf;
	for k = 2 : numNodes
		do
			pos(k,:) = [randi(area(1)), randi(area(2))];
			d(k,:) = sqrt((pos(:,1) - pos(k,1)).^2 + (pos(:,2) - pos(k,2)).^2);
			if (min(d(k,1:k-1)) > dmax)
				[dd,i] = min(d(k,1:k-1));
				dd = 1 - dmax / dd;
				delta = dd * (pos(i,:) - pos(k,:));
				pos(k,:) += sign(delta) .* ceil(abs(delta));
				d(k,:) = sqrt((pos(:,1) - pos(k,1)).^2 + (pos(:,2) - pos(k,2)).^2);
			endif
			d(k,k) = Inf;
		until (min(d(k,1:k)) >= dmin)
	endfor	
	testbed.nodes = pos;

	% generate link matrix
	d = tril(d) + tril(d, -1)';
	testbed.linkMatrix = testbed.linkModel.d2g(d);
	testbed.linkMatrix_dB = 10 * log10(testbed.linkMatrix);

endfunction

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function testbed = importFlocklab(xmlFile, linkModel, options = struct())

	testbed = struct();
	testbed.linkModel		= linkModel;
	testbed.generateOptions	= options;
	testbed.dmax 			= NA;

	f = fopen(xmlFile, 'r');
	s = fgetl(f);
	s = fgetl(f);
	fclose(f);

	[S, E, TE, M, T, NM, SP] = regexp(s, '<link src="(\d+)" dest="(\d+)" prr="([^"]+)"[^/]*/>');

	% generate nodes
	nodes = [];
	for i = 1 : length(T)
		d = str2double(T{i});
		nodes = union(nodes, d(1:2));
	endfor
	testbed.nodes = [1 : length(nodes); nodes']';
	testbed.area = [length(nodes) + 1, max(nodes) + 1];

	% generate link matrix
	p = [];
	d = Inf(length(nodes));
	for i = 1 : length(T)
		link = str2double(T{i});
%		p(lookup(nodes, link(2)), lookup(nodes, link(1))) = link(3);
		d(lookup(nodes, link(2)), lookup(nodes, link(1))) = p2d(linkModel, link(3));
	endfor
%	d = (d + d') / 2;
	testbed.linkMatrix = testbed.linkModel.d2g(d);
	testbed.linkMatrix_dB = 10 * log10(testbed.linkMatrix);
	
endfunction

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% link characteristics
linkModel 				= struct();
linkModel.Ptx_dBm 		= 0;
linkModel.Pnoise_dBm	= -90;
linkModel.SINR2p		= @(SINR_dB) 1 ./ (1 + exp(-SINR_dB + 5));
linkModel.d2g 			= @(d) ((3e8 / 2.4e9) / (4 * pi))^2 ./ d.^4;
linkModel.g2d 			= @(g) nthroot(((3e8 / 2.4e9) / (4 * pi))^2 ./ g, 4);

action = menu('choose action:', 'generate testbed(s)', 'import FlockLab', 'show testbed', 'show link model');
switch (action)

	case 1
		params = inputdlg( ...
			{'area', 'numNodes', 'options (key=val;key=val;...)', 'numTestbeds', 'fileNameTemplate'}, ...
			'testbed parameters', 1, ...
			{'50,50', '8', 'dmin=sqrt(2);dmax.p=0.9', '1', 'data/testbed_%d.dat'} ...
		);
		if (length(params) > 0)

			area 			= eval(sprintf("[%s]", params{1}));
			numNodes 		= eval(params{2});
			o				= params{3};
			numTestbeds		= eval(params{4});
			fileNameTemplate= params{5};

			options = struct();
			o = strsplit(o, ';');
			for i = 1 : length(o)
				eval(['options.' o{i} ';']);
			endfor

			for i = 1 : numTestbeds
				printf("generating testbed %d ...\n", i); fflush(stdout);
				testbed = generateTestbed(linkModel, area, numNodes, options);
				if (!isempty(fileNameTemplate))
					save(sprintf(fileNameTemplate, i), 'testbed');
				endif
			endfor
			printf("done.\n");

		endif

	case 2
		[fileName, path] = uigetfile('data/*.xml', 'select FlockLab link_feed file');
		if (fileName)
			xmlFile = [path fileName];
			[fileName, path] = uiputfile('data/*.dat', 'Save testbed file as');
			printf("generating testbed from %s...\n", xmlFile); fflush(stdout);
			testbed = importFlocklab(xmlFile, linkModel);
			if (fileName)
				printf("saving testbed to %s...\n", [path fileName]); fflush(stdout);
				save([path fileName], 'testbed');
			endif
			printf("done.\n");
		endif
		
	case 3
		[fileName, path] = uigetfile('data/*.dat', 'select testbed');
		if (fileName)
			load([path fileName], 'testbed');

			[~, name] = fileparts(fileName);
			[videoFile, path] = uiputfile(['data/' name '.gif'], 'Save critical path animation as');
			if (videoFile)
				videoFile = fullfile(path, videoFile);
			endif
		else
			fileName = "";
			videoFile = "";
		endif

		pos = testbed.nodes;
		if (isfield(testbed, 'area'))
			area = testbed.area;
		else
			area = max(pos);
		endif

		dmax = [
			testbed.dmax
			p2d(testbed.linkModel, 0.5)
			p2d(testbed.linkModel, 1 - d2p(testbed.linkModel, testbed.dmax))
		]

		% create colormap
		% for some reason colorcube produces grayscale if N <= 8 -> circumvent this
		% for some reason black ellipses are interpreted as transparent when saving to png -> remove black
		% remove white since it is the background color
		cmap = colorcube(max(rows(pos) + 2, 9));
		cmap = cmap(find(max(cmap, [], 2)), :);
		cmap = cmap(find(min(cmap-1, [], 2)), :);
		cmap = cmap(1 : rows(pos), :);

		figure('name', fileName); clf();
		scatter(pos(:,1), pos(:,2), 400, cmap); 
%		xlim([0, max(area) + 1]); ylim(xlim());
		axis([0, area(1), 0, area(2)]);
		grid on;
		t = strtrim(cellstr(num2str([1 : rows(pos)]')));
		text(pos(:,1), pos(:,2), t, "horizontalalignment", "center");
		title(fileName, 'interpreter', 'none');

		offset = get(gca(), 'position');
		limits = axis();
		scale = [offset(3) / (limits(2) - limits(1)), offset(4) / (limits(4) - limits(3))];
		r = dmax * scale;
		annoPos = [];
		for i = 1 : rows(pos)
			x = offset(1) + pos(i,1) * scale(1);
			y = offset(2) + pos(i,2) * scale(2);
			annoPos(i,1:2) = [x, y];
		endfor

		d = zeros(rows(pos));
		for i = 1 : rows(pos)
			for k = i+1 : rows(pos)
        % l2 norm is the distance between two positions
				d(i,k) = norm(pos(i,:) - pos(k,:), 2);
			endfor
		endfor

		r_dBm = testbed.linkModel.Ptx_dBm + 10 * log10(testbed.linkModel.d2g(d));
		SINR_dB = r_dBm - testbed.linkModel.Pnoise_dBm;
		p = testbed.linkModel.SINR2p(SINR_dB) .* (d <= testbed.dmax);
		p = eye(size(p)) + triu(p, 1) + triu(p, 1)';

		h = zeros(rows(pos));
		I = h;
		for i = 1 : rows(pos)
			t = d(i, i+1 : rows(pos)) <= testbed.dmax;
			h(i, i+1 : rows(pos)) = 1 + 1e9 * (1 - t);
			I(i, i+1 : rows(pos)) = (i+1 : columns(h)) .* t;
			I(i+1 : rows(pos), i) = i * t;
		endfor
		h += triu(h, 1)';
		I2 = I;
		p2 = p;

		old = [];
		it = 1;
		while (!isequal([h,p,p2], old))
			printf("\ranalyzing path statistics, iteration %d of (about) %d", it, round(sqrt(rows(pos))) + 1); fflush(stdout); it += 1;
			old = [h,p,p2];
			for i = 1 : rows(pos)
				for k = 1 : rows(pos)
					if (i == k)
						continue;
					endif

					t1 = [h(i,:); h(i,k) + h(k,:)];
					t2 = [p(i,:); p(i,k) * p(k,:)];
					[~, t] = min(t1 - t2);
					t = sub2ind([2, rows(pos)], t, [1:rows(pos)]);
					h(i,:) = t1(t);
					p(i,:) = t2(t);
					n = k;
					while (n > 0 && I(i,n) != n)
						n = I(i,n);
					endwhile
					I(i,:) = [I(i,:); repmat(n, 1, rows(pos))](t);

					t2 = [p2(i,:); p2(i,k) * p2(k,:)];
					[~, t] = max(t2);
					t = sub2ind([2, rows(pos)], t, [1:rows(pos)]);
					p2(i,:) = t2(t);
					n = k;
					while (n > 0 && I2(i,n) != n)
						n = I2(i,n);
					endwhile
					I2(i,:) = [I2(i,:); repmat(n, 1, rows(pos))](t);

%					[h(i,:), t] = min([h(i,:); h(i,k) + h(k,:)]);
%					t = sub2ind([2, rows(pos)], t, [1:rows(pos)]);
%					I(i,:) = [I(i,:); repmat(k, 1, rows(pos))](t);
				endfor
			endfor
		endwhile
		printf("\n");

		nodeDegree = zeros(rows(pos), 1);
		for i = 1 : rows(h)
			nodeDegree(i) = columns(h) - nnz(h(i,:) - 1);
		endfor
		
		if (videoFile)
			tempFile = [tempname() '.png'];
			images = uint8(zeros(0,0,0,0));
			addFrameCmd = [ ...
				"print(tempFile, '-S800,600', '-GraphicsAlphaBits=4');" ...
				"images(:,:,:,size(images, 4) + 1) = imread(tempFile);" ...
%				"imwrite(imread(tempFile), 'test.gif', 'writemode', 'append', 'DelayTime', 1, 'LoopCount', 1);" ...
				"delete(tempFile);" ...
			];
		else
			addFrameCmd = "";
		endif

%		set(gcf(), 'visible', 'off');

		eval(addFrameCmd);

		[t,i1] = max(h - p);
		[~,i2] = max(t);
		i1 = i1(i2);
		do
			n1 = i1;
			n2 = I(i1, i2);
			i1 = n2;
			annotation('ellipse', [annoPos(n1,:) - r(1,:), 2 * r(1,:)], 'linestyle', '--', 'edgecolor', cmap(n1,:));
			annotation('ellipse', [annoPos(n1,:) - r(2,:), 2 * r(2,:)], 'linestyle', '-.', 'edgecolor', cmap(n1,:));
			annotation('ellipse', [annoPos(n1,:) - r(3,:), 2 * r(3,:)], 'linestyle', ':', 'edgecolor', cmap(n1,:));
			eval(addFrameCmd);
			x = offset(1) + [pos(n1,1), pos(n2,1)] .* scale(1);
			y = offset(2) + [pos(n1,2), pos(n2,2)] .* scale(2);
			annotation('line', x, y, 'color', 'red');
			eval(addFrameCmd);
		until (i1 == i2);
		annotation('ellipse', [annoPos(n2,:) - r(1,:), 2 * r(1,:)], 'linestyle', '--', 'edgecolor', cmap(n2,:));
		annotation('ellipse', [annoPos(n2,:) - r(2,:), 2 * r(2,:)], 'linestyle', '-.', 'edgecolor', cmap(n2,:));
		annotation('ellipse', [annoPos(n2,:) - r(3,:), 2 * r(3,:)], 'linestyle', ':', 'edgecolor', cmap(n2,:));
		eval(addFrameCmd);

		[t,i1] = min(p2);
		[~,i2] = min(t);
		i1 = i1(i2);
		do
			n1 = i1;
			n2 = I2(i1, i2);
			i1 = n2;
			x = offset(1) + [pos(n1,1), pos(n2,1)] .* scale(1);
			y = offset(2) + [pos(n1,2), pos(n2,2)] .* scale(2);
			annotation('line', x, y, 'color', [.5 0 0], 'linestyle', '--');
		until (i1 == i2);
		eval(addFrameCmd);

		if (videoFile)
			imwrite(images, videoFile, 'DelayTime', 1, 'LoopCount', 1);
			%avi=avifile('test.avi', 'fps', 2);
			%for i = 1 : size(im,4)
			%	addframe(avi, double(im(:,:,:,i)) ./ 255);
			%endfor
		endif

%		set(gcf(), 'visible', 'on');

		figure('name', fileName); clf();
		bar(1 : max(max(h)), histc(vec(triu(h, 1)), 1 : max(max(h))));
		grid on;
		title([fileName ' #hops histogram'], 'interpreter', 'none');

		figure('name', fileName); clf();
		bar(1 : max(nodeDegree), histc(nodeDegree, 1 : max(nodeDegree)));
%		hist(nodeDegree, 1 : max(nodeDegree));
		grid on;
		title([fileName ' node degree histogram'], 'interpreter', 'none');

	case 4
		[fileName, path] = uigetfile('data/*.dat', 'select testbed');
		load([path fileName], 'testbed');

		figure('name', fileName); clf();
		d = 1 : max(max(testbed.nodes)) * sqrt(2) * 1.1;
		r = testbed.linkModel.Ptx_dBm + 10 * log10(testbed.linkModel.d2g(d));
		n = testbed.linkModel.Pnoise_dBm;

		subplot(2,2,1);
		plot(d, r, [d(1), d(end)], [n, n], ';noise floor;');
		title('Rx power vs. distance');
		xlabel('distance [m]'); ylabel('Rx power [dBm]');
		grid on;

		subplot(2,2,2);
		semilogx(d, r, [d(1), d(end)], [n, n], ';noise floor;');
		title('Rx power vs. distance');
		xlabel('distance [m]'); ylabel('Rx power [dBm]');
		grid on;

		subplot(2,2,3);
		x = -10:20;
		plot(x, testbed.linkModel.SINR2p(x));
		title('Rx probability vs. SINR');
		xlabel('SINR [dB]'); ylabel('Rx probability');
		ylim([0, 1]);
		grid on;

		subplot(2,2,4);
		plot(d, testbed.linkModel.SINR2p(r - n));
		title('Rx probability vs. distance w/o interference');
		xlabel('distance [m]'); ylabel('Rx probability');
		ylim([0, 1]);
		grid on;

endswitch

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
