//@author Sottithat Winyarat winyarat@seas.upenn.edu
#include "orb.h"

//======================================================================================
//===================== Image Preprocessing methods ====================================
//======================================================================================

//load 3x4 transformation matrices containing groundtruth poses
vector<Matrix<double, 3, 4, RowMajor>> orb::loadGroundTruths(const string& fname){
	Matrix<double, 3, 4, RowMajor> mat;
	vector<Matrix<double, 3, 4, RowMajor>> poses;
	ifstream myfile(fname);
	string line;
	if(myfile.is_open()){
		while (getline (myfile,line) ){
			istringstream iss(line);
			for(int i = 0; i< mat.size(); i++){
				iss >> mat(i);
			}
			poses.push_back(mat);
		}	
		myfile.close();
	}
	return poses;
}

void orb::readPNG(const string& filename, rawRGB& raw){
	unsigned decodeerror = lodepng::decode(raw.image, raw.width, raw.height, filename);
	if(decodeerror){
		cout << "decoder error " << decodeerror << ": " << lodepng_error_text(decodeerror) << std::endl;
	}
}

void orb::writePNG(string filename, const MatrixXd& grayMat){
	//lodepng encoder is rowmajor
	Matrix<double, Dynamic, Dynamic, RowMajor> final(grayMat);
	vector<unsigned char> out;
	int numPix = grayMat.cols()*grayMat.rows();
	for(int i = 0; i < numPix; i++){
		out.push_back( final(i) );
		out.push_back( final(i) );
		out.push_back( final(i) );
		out.push_back( 255 );
	}
	unsigned error = lodepng::encode(filename, out, grayMat.cols(), grayMat.rows());
	if(error) std::cout << "encoder error " << error << ": "<< lodepng_error_text(error) << std::endl;
}


//load and extract camera intrinsic parameters from a 3x4 projection matrix
//In KITTI dataset, camID == 0 for left camera and camID == 1 for right camera
//IMPORTANT REMARK: imscale indicates the proportion of stereo image size to the raw image size. 
map<string, double> orb::loadCamIntrinsics(string fname, int camID, double imscale){
	Matrix<double, Dynamic, Dynamic, RowMajor> proj(3,4);
	ifstream file(fname);
	string projID = "P" + to_string(camID);
	string line;
	//find the line corresponding to a given camera ID
	if(file.is_open()){
		while ( getline (file,line) ){
			if(projID == line.substr(0,2)){
				line = line.substr(line.find_first_of(":")+1);
				break;
			}	  
		}
		file.close();
	}
	else{
		cerr << "load camerca intrinsic failed.\n" ;
	}


	//read in projection matrix
	istringstream iss(line);
	for(int i = 0; i< proj.size(); i++){
		iss >> proj(i);
	}

	//build and return intrinsics map
	map<string, double> out = {{"fx",proj(0,0)*imscale},{"fy",proj(1,1)*imscale},{"cx",proj(0,2)*imscale},{"cy",proj(1,2)*imscale},{"bline",-proj(0,3)*imscale/proj(0,0)}};
	return out;		
}

// ****** IMPORTANT ****** the output matrix must have ROWMAJOR for correct PNG encoding.
//first input is a raster vector of RGBa pixel values  
MatrixIM orb::rbg2grayMat(const vector<unsigned char>& image, int width, int height){
	vector<double> newimage;
	int numPix = width * height;
	for(int i = 0; i < numPix; i++){
		int j = 4*i;
		double intensity = (image[j]*0.299 + image[j+1]*0.587 + image[j+2]*0.114);
		newimage.push_back(intensity);
	}
  	//copy construct Eigen Matrix from vector of intensity values;  	
	MatrixIM matGray = Eigen::Map<MatrixIM>(newimage.data(), height, width);
	return matGray;
}


//1D Convolution with odd-length kernel
Eigen::MatrixXd orb::conv1d(const Eigen::MatrixXd& baseIm, const Eigen::MatrixXd& kernel){
	//0. setup output and kernel properties
	int kernelLength = kernel.size();
	if(!(kernelLength % 2)){
		cerr << "1D Convolution Kernel must be of odd length. \n";
		abort();
	}
	bool isVertKernel = (kernel.cols() == 1)? true:false;
	int kernelHalfWidth = kernelLength/2;
	int baseRows = baseIm.rows();
	int baseCols = baseIm.cols();
	MatrixXd out(baseRows,baseCols);


	//1. convolve base image with vertical kernel
	if(isVertKernel){		
		MatrixXd padded = MatrixXd::Zero(2*kernelHalfWidth + baseRows, baseCols);
		padded.block(kernelHalfWidth, 0, baseRows,baseCols) = baseIm;
		for(int i = 0; i < baseRows; i++){
			for(int j = 0; j < baseCols; j++){
				out(i,j) = (kernel.cwiseProduct(padded.block(i,j, kernel.rows(), kernel.cols()))).sum();				
			}
		}
	}
	//2. convolve with horizontal kernel
	else{
		MatrixXd padded = MatrixXd::Zero(baseRows, 2*kernelHalfWidth + baseCols);
		padded.block(0,kernelHalfWidth, baseRows,baseCols) = baseIm;
		for(int i = 0; i < baseRows; i++){
			for(int j = 0; j < baseCols; j++){
				out(i,j) = (kernel.cwiseProduct(padded.block(i,j, kernel.rows(), kernel.cols()))).sum(); 				
			}
		}
	}

	return out;
}

//2D convolution with seperable and symmetric Kernel; eg. Gaussian
Eigen::MatrixXd orb::conv2d_separable(const Eigen::MatrixXd& baseIm, const Eigen::MatrixXd& oneDKernel){
	return conv1d(conv1d(baseIm, oneDKernel), oneDKernel.transpose());
}



MatrixXd orb::conv2d(const MatrixIM& baseIm, const MatrixXd& kernel){
	//kernel must be of odd-width and height
	int baseRows = baseIm.rows();
	int baseCols = baseIm.cols();
	int krows = kernel.rows();
	int kcols = kernel.cols();
	MatrixXd out(baseRows,baseCols);
	
	//1. zero-pad base image
	MatrixXd padded = MatrixXd::Zero(2*(krows/2) + baseRows, 2*(kcols/2) + baseCols);
	padded.block(krows/2, kcols/2, baseRows,baseCols) = baseIm;

	//2. compute convolved signals.
	double temp = 0;
	for(int i = 0; i < baseRows; i++){
		for(int j = 0; j < baseCols; j++){
			temp = (kernel.cwiseProduct(padded.block(i,j,krows,kcols))).sum();		
			out(i,j) = temp;
		}
	}
	return out;
}


//build a 2D Gaussian Kernel
MatrixXd orb::gaussianKernel2D(int width, double sigma){
	if( width %2 == 0 ){
		cerr << "Kernel must have odd width. \n" ;
		abort();
	}

	MatrixXd out(width,width);
    double r, s = 2.0 * sigma * sigma;
  	int halfwidth = width/2;
    for (int x = -halfwidth; x <= halfwidth; x++) {
        for (int y = -halfwidth; y <= halfwidth; y++) {
            r = x * x + y * y;
            out(x+halfwidth, y+halfwidth) = (exp(-r/ s)) / (M_PI * s);
        }
    }
    return out/out.sum();
}

//reduce the source image resolution by 1/2 along each axis
MatrixXd orb::subsample(MatrixXd& source){
	auto rinds = VectorXi::LinSpaced(ceil(source.rows()/2.0), 0, source.rows());
	auto cinds = VectorXi::LinSpaced(ceil(source.cols()/2.0), 0, source.cols());
	return source(rinds, cinds);
}




//================================================================================
//===================== FAST detector methods ====================================
//================================================================================

//build pixel offsets to the neighborhood ring of a given radius
vector<orb::pixelCoord> orb::getNeighborOffsets(const int& radius){
	vector<orb::pixelCoord> out;

	int numElements = 4*M_PI*(radius); //number of samples
	auto angles = VectorXd::LinSpaced(numElements, 0,  2*M_PI );
	auto aIter = angles.begin();
	
	int dv_prev = 0;
	int du_prev = 0;
	int dv = 0;
	int du = 0;
	double theta = 0;
	while(aIter != angles.end()){
		theta = *aIter;
		if(aIter <= angles.begin() + numElements/8){
			dv = (sin(theta)*(radius));
			du = ceil(cos(theta)*(radius));
		}
		else if(aIter <= angles.begin() + 2*numElements/8){
			dv = ceil(sin(theta)*(radius));
			du = ceil(cos(theta)*(radius));
		}
		else if(aIter <= angles.begin() + 3*numElements/8){
			dv = ceil(sin(theta)*(radius));
			du = (cos(theta)*(radius));
		}
		else if(aIter <= angles.begin() + 4*numElements/8){
			dv = ceil(sin(theta)*(radius));
			du = floor(cos(theta)*(radius));
		}
		else if(aIter <= angles.begin() + 5*numElements/8){
			dv = (sin(theta)*(radius));
			du = floor(cos(theta)*(radius));
		}
		else if(aIter <= angles.begin() + 6*numElements/8){
			dv = floor(sin(theta)*(radius));
			du = floor(cos(theta)*(radius));
		}
		else if(aIter <= angles.begin() + 7*numElements/8){
			dv = floor(sin(theta)*(radius));
			du = (cos(theta)*(radius));
		}
		else{
			dv = floor(sin(theta)*(radius));
			du = ceil(cos(theta)*(radius));
		}
		
		//check for repeated (du,dv)
		if(dv != dv_prev || du != du_prev){
			dv_prev = dv;
			du_prev = du;
			out.push_back(orb::pixelCoord(du,dv));	
		}
		aIter++;
	}//end while

	return out;
}

//Compare anchors against the center pixel at (u,v)
bool orb::testAnchors(MatrixXd& image, const int& u, const int& v, vector<int>& anchors, const double& FASTthreshold, vector<orb::pixelCoord>& offsets, bool isLess, int quater){	
	double centerValue  = image(v,u);
	auto aIter = anchors.begin();
	while(aIter != anchors.end()){
		//test dimmer neighbors
		if(isLess){
			if(! (image(v + offsets[*aIter].v, u + offsets[*aIter].u) < centerValue - FASTthreshold)){
			 	aIter = anchors.erase(aIter);
			}
			else{
				aIter++;
			}
		}
		
		//test brigher neighbors
		else{
			if(! (image(v + offsets[*aIter].v, u + offsets[*aIter].u) > centerValue + FASTthreshold)){
			 	aIter = anchors.erase(aIter);
			}
			else{
				aIter++;
			}
		}

		//If only 2 non-consecutive anchors remain, test fails
		if(anchors.size() == 2 && abs(anchors[1] - anchors[0]) > quater ){
			return false;
		}			
	}	

	return true;
}


//Test for a FAST feature. 
//isLess:  true when testing for neighbors dimmer than center pixel
bool orb::FASTtest(MatrixXd& image, const int& u, const int& v, const double& FASTthreshold, vector<orb::pixelCoord>& offsets, bool isLess){	
	int numberOfNeighbors = offsets.size();
	int quater = numberOfNeighbors/4;

	// Anchors are initially set to east, north, west, and south neighbor indices. 
	// If anchors pass preliminary test, then rotate anchors ccw/cw and test the next set of 3 neighbors.
	// If they pass, continue in the same fashion until at least 2 quaters of all neighbors have been tested. If they do not pass, return false.

	//1. preliminary test on North and South anchors
	double centerValue = image(v,u);
	if( abs(image(v + offsets[0].v, u + offsets[0].u)  - centerValue) < FASTthreshold && abs(image(v + offsets[2*quater].v, u + offsets[2*quater].u) - centerValue) < FASTthreshold){
		return false;
	}
	if( abs(image(v + offsets[1].v, u + offsets[1].u)  - centerValue) < FASTthreshold && abs(image(v + offsets[3*quater].v, u + offsets[3*quater].u) - centerValue) < FASTthreshold){
		return false;
	}


	//2. Initialize anchors to N,S,E,W indices
	vector<int> anchors;
	anchors.push_back(0); //east
	anchors.push_back(1*quater); //north
	anchors.push_back(2*quater); //west
	anchors.push_back(3*quater); //south

	//3. secondary test
	//3.1 rotate anchors clockwise and test them until at least 2 quaters of neighbors pass	
	int qtemp = 1;
	vector<int> cwAnchors(anchors);
	while(qtemp < quater){
		if(! testAnchors(image, u,v, cwAnchors, FASTthreshold, offsets, isLess, quater)){
			break;
		}	
		for(auto& a: cwAnchors){
			a = (a+1) % numberOfNeighbors;
		}
		qtemp++;
	}

	//3.2 rotate anchors counter-clockwise and test them until at least 2 quaters of neighbors pass
	qtemp = 1;
	vector<int> ccwAnchors(anchors);
	while(qtemp < quater){
		if(! testAnchors(image, u,v,ccwAnchors, FASTthreshold, offsets, isLess, quater)){
			return false;
		}	
		for(auto& a: ccwAnchors){
			a = (((a-1) % numberOfNeighbors) + numberOfNeighbors) % numberOfNeighbors;
		}
		qtemp++;
	}

	return true;
}


//extract FASTfeatures from image using a FAST ring defined by radius.
//FASTthreshold: tolerance on center-vs-neighbor intensity difference.
//N: the target number of FAST features to extract
void orb::extractFASTfeats(vector<orb::FASTfeature>& feats, MatrixXd& image, const double imscale, const int FASTradius, const double FASTthreshold){
	int numElements  = 0;
	int rows = image.rows();
	int cols = image.cols();
	int numPixs = rows*cols;
	int patchWidth = 2*FASTradius+1;
	int patchGuard = 3*FASTradius;

	vector<orb::pixelCoord> neighborOffsets = orb::getNeighborOffsets(FASTradius);
	//Sweep a square window of width 2*FASTradius over the image. 
	//Find FAST features and suppress non-maxima within each window.
	for(int v = FASTradius; v < rows - patchGuard; v += patchWidth ){
		for(int u = FASTradius; u < cols- patchGuard; u += patchWidth){
			orb::FASTfeature ffeat = suppressNonMax(image, u, v, FASTradius, FASTthreshold, imscale, neighborOffsets);
			//reject feature at (0,0)
			if(ffeat.v != 0 && ffeat.u != 0){
				// image(ffeat.v, ffeat.u) = 255; // for DEBUGGING: mark feature white 
				feats.push_back(ffeat);
			}
			
		}
	}
}

//sum of abosolute differences between center pixel and its neighbors
double orb::SAD(MatrixXd& image, int u, int v, int patchWidth){
	int halfwidth = patchWidth/2;
	double centerValue = image(v,u);
	auto temp = MatrixXd::Ones(patchWidth,patchWidth)*centerValue;
	auto diff = image.block(v-halfwidth,u-halfwidth,patchWidth,patchWidth) - temp;
	return diff.cwiseAbs().sum();
}

//compute the Harris corner measure of a patch centered at (u,v)
//reference: Szeliski pages 185-189
double orb::harrisMeasure(MatrixXd& image, int u, int v, int patchWidth, const MatrixXd& weight, double alpha){
	//0. Sobel approximations to Gaussian derivatives
	MatrixXd Sx(3,3);
	MatrixXd Sy(3,3);
	Sx << -1, 0, 1,
		  -2, 0, 2,
		  -1, 0, 1;
	Sy << 1, 2, 1,
		  0, 0, 0,
		 -1, -2, -1;
	Sy = Sy/8.0;

	//1. Compute patch gradients
	int halfwidth = floor(patchWidth/2);
	MatrixXd Ix = orb::conv2d(image.block(v-halfwidth,u-halfwidth,patchWidth,patchWidth),  Sx);
	MatrixXd Iy = orb::conv2d(image.block(v-halfwidth,u-halfwidth,patchWidth,patchWidth),  Sy);

	//2. Compute Elements of Hessian
	MatrixXd Ixx = Ix.cwiseProduct(Ix);
	MatrixXd Ixy = Ix.cwiseProduct(Iy);
	MatrixXd Iyy = Iy.cwiseProduct(Iy);
	double ixx = Ixx.cwiseProduct(weight).sum();
	double ixy  = Ixy.cwiseProduct(weight).sum();
	double iyy  = Iyy.cwiseProduct(weight).sum();

	return  ixx*iyy - ixy*ixy - alpha*(ixx+iyy)*(ixx+iyy);
}

//detect FAST features in a window of width FASTradius. Based on harris measure, suppress non-maxima in this window.
//(u,v): top left corner of window
orb::FASTfeature orb::suppressNonMax(MatrixXd& image, int u, int v, int FASTradius, double FASTthreshold, double imscale, vector<orb::pixelCoord>& offsets){
	orb::FASTfeature maxFeature;
	double curMaxSad = -std::numeric_limits<double>::max();
	double sad = 0;
	int patchWidth = 2*FASTradius+1;
	auto gaussianWindow = orb::gaussianKernel2D(patchWidth, sqrt(2));

	for(int y = v; y < v+ patchWidth; y++){
		for(int x = u; x < u + patchWidth; x++){
			if(FASTtest(image, x,y, FASTthreshold, offsets, true) || FASTtest(image, x,y, FASTthreshold, offsets, false)){
				sad = orb::SAD(image,x,y, patchWidth);	
				if(sad > curMaxSad){
					maxFeature = orb::FASTfeature(x, y, FASTradius, imscale, 0.0, 0.0);
					curMaxSad = sad;
				}
			}
		}
	}

	//if max feature is detected, compute its harris measure and orientation
	if(maxFeature.u != 0 && maxFeature.v != 0 ){
		maxFeature.harris = orb::harrisMeasure(image, maxFeature.u, maxFeature.v, patchWidth, gaussianWindow, 0.04);
		featureOrientationID(image, maxFeature);
	}
	return maxFeature;
}


//compute the orientation(in radian) of a feature using the intensity centroid technique
void orb::featureOrientationID(MatrixXd& image, orb::FASTfeature& feat){
	int radius = feat.radius;
	VectorXd displacement = VectorXd::LinSpaced(2*radius+1, -radius, radius); //displacement coords from patch's center
	MatrixXd patch = image.block(feat.v - radius, feat.u - radius, 2*radius+1, 2*radius+1);
	double xMoment = (patch*displacement).sum();
	double yMoment = ((patch.transpose())*displacement.reverse()).sum(); //reverse displacement vector since v-axis points downward in pixel-space 
	double rad = atan2(yMoment, xMoment);
	
	VectorXd degrees = VectorXd::LinSpaced(30, 0, 2*M_PI*29/30); //30 discretized angle bins. 
	rad = rad < 0? 2*M_PI + rad: rad; // rad in [0, 2pi]
	
	//assign feature's orientation ID to be one that is closest to any of the 30 descritized angles;
	auto temp = degrees.array() - rad;
	auto tempabs = temp.cwiseAbs();
	int ID = 0;
	tempabs.minCoeff(&ID);
	feat.orientationID = ID; 
}


//draw white crosses over features on image
void orb::drawFeatures(MatrixXd& image, vector<orb::FASTfeature> ffeats){
	for(auto& f: ffeats){
		int u = f.u/f.scale;
		int v = f.v/f.scale;
		for(int i = u - f.radius/f.scale; i < u + f.radius/f.scale +1; i++){
			image(v,i) = 255;
		}
		for(int j = v - f.radius/f.scale; j < v + f.radius/f.scale +1; j++){
			image(j,u) = 255;
		}
	}	
}

void orb::drawStereoCorr(MatrixXd& leftimage, MatrixXd& rightimage, orb::StereoKeypt corr){
	int lower = floor(corr.FASTradius/(2.0));
	int upper = ceil(corr.FASTradius/(2.0));
	for(int i = corr.lpix.u - lower; i < corr.lpix.u + upper; i++){
		leftimage(corr.lpix.v,i) = 255;
	}
	for(int j = corr.lpix.v - lower; j < corr.lpix.v + upper; j++){
		leftimage(j,corr.lpix.u) = 255;
	}

	for(int i = corr.rpix.u - lower; i < corr.rpix.u + upper; i++){
		rightimage(corr.rpix.v,i) = 255;
	}
	for(int j = corr.rpix.v - lower; j < corr.rpix.v + upper; j++){
		rightimage(j,corr.rpix.u) = 255;
	}
}


void orb::drawTemporalCorr(MatrixXd& leftimage, MatrixXd& rightimage, vector<orb::TemporalCorrespondence> corrs){
	int lower;
	int upper;
	int imageWidth =  leftimage.cols();
	int imageHeight = leftimage.rows();
	for(auto& corr: corrs){
		lower = floor(corr.current.FASTradius/(2.0));
		lower = lower/corr.current.scale;
		upper = ceil(corr.current.FASTradius/(2.0));
		upper = upper/corr.current.scale;
		
		for(int i = max(corr.current.lpix.u - lower, 0); i < min(corr.current.lpix.u + upper, imageWidth ); i++){
			leftimage(corr.current.lpix.v,i) = 255;
		}
		for(int j = max(corr.current.lpix.v - lower,0); j < min(corr.current.lpix.v + upper, imageHeight); j++){
			leftimage(j,corr.current.lpix.u) = 255;
		}
	
		for(int i = max(corr.next.lpix.u - lower,0); i < min(corr.next.lpix.u + upper, imageWidth); i++){
			rightimage(corr.next.lpix.v,i) = 255;
		}
		for(int j = max(corr.next.lpix.v - lower,0); j < min(corr.next.lpix.v + upper,imageHeight); j++){
			rightimage(j,corr.next.lpix.u) = 255;
		}
	}
}


//================================================================================
//===================== BRIEF descriptor methods =================================
//================================================================================

//generate N binary-test pixel pairs drawn from a Gaussian distribution defined by sigma over a feature patch.
//The mothod returns a tuple of 2 Eigen::MatrixXi's, where the first matrix contains
//horizontally stacked x-y coordinates test points; 
//the second matrix contains x-y coordinates of the corresponding test points.
tuple<MatrixXd, MatrixXd> orb::generateTestPairs(int FASTradius, int N){
	
	MatrixXd left(2, N);
	MatrixXd right(2, N);
	default_random_engine generator;
  	normal_distribution<double> distribution(0,FASTradius/2.0);
  	double clamphigh = sqrt(2)*FASTradius/2;
  	double clamplow = -clamphigh;
  	double temp;

  	//clamping each test points so that after rotated, they remain valid within their feature patch and image
  	for(int i = 0; i < N; i++){
  		temp = clamp(distribution(generator), clamplow, clamphigh);
  		left(0, i) = int(temp);
  		temp = clamp(distribution(generator), clamplow, clamphigh);
  		left(1, i) = int(temp);
  		temp = clamp(distribution(generator), clamplow, clamphigh);
  		right(0, i) = int(temp);
  		temp = clamp(distribution(generator), clamplow, clamphigh);
  		right(1, i) = int(temp);
  	}

  	tuple<MatrixXd, MatrixXd> out(left, right);
  	return out;
}



//build a lookup table of rotated BRIEF test pairs
//key: angle bin number. Each bin is of 30 degrees increments. value: tuple of rotated test pairs
//leftTests and rightTests: Gaussian-generated test points
map<int, tuple<MatrixXi, MatrixXi>> orb::buildTestPairsLookup(MatrixXd& leftTests, MatrixXd& rightTests){
	map<int, tuple<MatrixXi, MatrixXi>> out;
	Matrix2d Rotation(2,2);
	
	auto degrees = VectorXd::LinSpaced(30, 0, 2*M_PI*29/30); //30 discretized degrees. 
	int key = 0;
	for(auto d: degrees){
		 Rotation(0,0) = cos(d);
		 Rotation(0,1) = -sin(d);
		 Rotation(1,0) = sin(d);
		 Rotation(1,1) = cos(d);
		 MatrixXi rotatedLeftTests = (Rotation * leftTests).cast<int>();
		 MatrixXi rotatedRightTests = (Rotation * rightTests).cast<int>();
		 out[key] = tuple<MatrixXi, MatrixXi>(rotatedLeftTests, rotatedRightTests);
		 key++;
	}
	return out;
}

//write feature patch as a PNG.
void orb::featurePatch2PNG(MatrixXd& image, orb::FASTfeature feat){
	int temp = 2*feat.radius + 1;
	writePNG("feat"+ to_string(feat.u) + to_string(feat.v) + ".png", image.block(feat.v - feat.radius, feat.u -  feat.radius,temp, temp));
}	

//generate BRIEF descriptors of FAST features
void orb::BRIEFdescriptors(vector<orb::FASTfeature>& feats, MatrixXd& image,  map<int, tuple<MatrixXi, MatrixXi>>& lookupTable){
	VectorXi center(2,1);

	for(auto& feat: feats){
		//1. retrieve rotated test point pairs according to feature's orientationID from a lookup table.	
		center(0,0) = feat.u;
		center(1,0) = feat.v;
		auto leftTests = get<0>(lookupTable[feat.orientationID]);
		auto rightTests = get<1>(lookupTable[feat.orientationID]);

		leftTests = leftTests.colwise() + center; //recenter test points
		rightTests = rightTests.colwise() + center;

		//2. build and assign a binary descriptor
		bitset<orb::BITNUMBER> descriptor;
		bool bin;
		for(int i = 0; i < BITNUMBER; i++){
			descriptor[i] = (image(leftTests(1,i), leftTests(0,i)) < image(rightTests(1,i), rightTests(0,i)));
		}
		
		feat.BRIEFdescriptor = descriptor;
	}	
} 	

int orb::hammingDistance(bitset<BITNUMBER> a, bitset<BITNUMBER> b){
	bitset<BITNUMBER> temp = a ^ b;
   	return temp.count();
}


//For every left feature, find a stereo correpondence in the right image.
//window: the row-wise offset above and below the current left pixel that a potential match on the right image could fall within. 
vector<orb::StereoKeypt> orb::stereoMatch(vector<orb::FASTfeature>& leftFeatures, vector<orb::FASTfeature>& rightFeatures, int window, map<string, double>& camIntrinsics){
	vector<orb::StereoKeypt> out;
	
	//1. Hash right image features into row-based bins
	map<int, vector<orb::FASTfeature>> rightBins; 
	for(auto f:rightFeatures){
		rightBins[f.v].push_back(f);
	}
	
	//2. For every left feature, search within rowwise window for a match in the right image
	std::sort(leftFeatures.begin(), leftFeatures.end(), [](auto const& a, auto const& b){ return a.v < b.v;});
	int lv = -1;
	vector<orb::FASTfeature> rightCandidates;
	for(auto lfeat: leftFeatures){
		//2.1 gather all potential candidates from the right image for each new row.
		if(lfeat.v != lv){
			lv = lfeat.v;
			rightCandidates.clear();
			for(int j = lv-window; j <= lv+window ; j++){
				if(rightBins.find(j) != rightBins.end()){
					rightCandidates.insert(rightCandidates.end(), rightBins[j].begin(), rightBins[j].end());
				}
			}
		}

		//2.2 compute hamming distance between a left feature and every right image candidate
		if(! rightCandidates.empty()){
			for(auto& r: rightCandidates){
				r.hdistance = orb::hammingDistance(lfeat.BRIEFdescriptor, r.BRIEFdescriptor);
			}
			//heapify rightCandidates and assign stereo correspondence
			std::make_heap(rightCandidates.begin(), rightCandidates.end(), [](auto lhs, auto rhs){return lhs.hdistance > rhs.hdistance;});
			auto rfeat = rightCandidates.front();
			//registering stereo correspondence only for features whose binary descriptors are at most (1/4)*BITNUMBER apart and with non-negative disparity
			if(rfeat.hdistance < orb::BITNUMBER/2.0 && (lfeat.u - rfeat.u) > 0 ){
				out.push_back(orb::StereoKeypt(pixelCoord(int(lfeat.u/lfeat.scale), int(lfeat.v/lfeat.scale)), pixelCoord(int(rfeat.u/rfeat.scale), int(rfeat.v/rfeat.scale)), camIntrinsics, lfeat.BRIEFdescriptor, lfeat.radius, lfeat.scale));
				//pop matched right image feature
				pop_heap(rightCandidates.begin(), rightCandidates.end(),  [](auto lhs, auto rhs){return lhs.hdistance > rhs.hdistance;});
				rightCandidates.pop_back();
				//remove matched right feature from rightBins
				rightBins[rfeat.v].erase(std::remove_if(rightBins[rfeat.v].begin(), 
                              rightBins[rfeat.v].end(),
                              [lfeat](orb::FASTfeature x){return x.u == lfeat.u;}), rightBins[rfeat.v].end());
			}

		}
	}

	return out;
}

//match stereo keypoints across 2 temporal image frames, searching horizontally within a given window
vector<orb::TemporalCorrespondence>  orb::temporalMatch(vector<orb::StereoKeypt>& currKeypts, vector<orb::StereoKeypt>& nextKeypts, int window){
	int matchNum = 0;

	//1. Hash next-frame keypts into row-based bins
	map<int, vector<orb::StereoKeypt>> nextBins; 
	for(auto k:nextKeypts){
		nextBins[k.lpix.v].push_back(k);
	}

	//2. For every current-frame keypoint, search for a potention match in the next frame that lies within a given row-window
	std::sort(currKeypts.begin(), currKeypts.end(), [](auto const& a, auto const& b){ return a.lpix.v < b.lpix.v;});
	int currRow = -1;
	vector<orb::StereoKeypt> nextFrameCands; 	
	vector<orb::TemporalCorrespondence> out;
	for(auto curr: currKeypts){
		//2.1 gather all candidates from the next frame
		if(curr.lpix.v != currRow){
			currRow = curr.lpix.v;
			nextFrameCands.clear();
			for(int j = currRow-(window/2); j <= currRow+window ; j++){
				if(nextBins.find(j) != nextBins.end()){
					nextFrameCands.insert(nextFrameCands.end(), nextBins[j].begin(), nextBins[j].end());
				}
			}
		}
		//2.2 compute the hamming distance between curr keypoint and all next-frame candidates 
		if(! nextFrameCands.empty()){
			for(auto& n: nextFrameCands){
				n.hdistance = orb::hammingDistance(curr.BRIEFdescriptor, n.BRIEFdescriptor);
			}
			//heapify next-frame Candidates and assign temporal correspondence
			std::make_heap(nextFrameCands.begin(), nextFrameCands.end(), [](auto lhs, auto rhs){return lhs.hdistance > rhs.hdistance;});
			auto nextKey = nextFrameCands.front();
			//registering stereo pairs whose binary descriptors are at most (1/4)*BITNUMBER apart. 
			if(nextKey.hdistance < orb::BITNUMBER/4.0){					
				out.push_back(orb::TemporalCorrespondence(curr, nextKey, 1- nextKey.hdistance/double(orb::BITNUMBER)));
				//pop matched next-frame keypoint
				pop_heap(nextFrameCands.begin(), nextFrameCands.end(),  [](auto lhs, auto rhs){return lhs.hdistance > rhs.hdistance;});
				nextFrameCands.pop_back();
				//remove matched right feature from rightBins
				nextBins[nextKey.lpix.v].erase(std::remove_if(nextBins[nextKey.lpix.v].begin(), 
                              nextBins[nextKey.lpix.v].end(),
                              [nextKey](orb::StereoKeypt x){return x.lpix.u == nextKey.lpix.u;}), nextBins[nextKey.lpix.v].end());
			}
		}
	}//end for-loop over all current keypts
	orb::normalizeCorrespondenceWeight(out);
	return out;
}

void orb::normalizeCorrespondenceWeight(vector<orb::TemporalCorrespondence>& corrs){
	double norm = std::accumulate(corrs.begin(), corrs.end(), 0.0, [&](double sum, const auto& a){return sum + a.weight ; });
	for(auto c:corrs){
		c.weight = c.weight/norm;
	}
}


vector<vector<double>> orb::correspondences2scenePoints(vector<orb::TemporalCorrespondence>& corrs){
	vector<vector<double>> out;
	for(auto c: corrs){
		vector<double> temp{c.next.X , c.next.Y, c.next.Z};
		out.push_back(temp);
	}
	return out;
} 


