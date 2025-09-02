/**
 * @file main.cpp
 * @brief Program pro lokalizaci řezu pro Array Tommography.
 * @author Marek Konečný (xkonec86)
 * @date 2025-04-29
*/

#include <opencv2/opencv.hpp>
#include <omp.h>
#include <random>
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
//#include <chrono>

// ------------------------- pomocné struktury ----------------------------
struct LineABC { double a{}, b{}, c{}; }; // normalizovaná přímka Ax + By + C = 0
struct CornerSet { cv::Point tl, tr, br, bl; };
//using Clock = std::chrono::high_resolution_clock;
std::default_random_engine rng;

// --------------------- pomocé geometrické funkce ------------------------
inline LineABC line_params(const cv::Point& p1, const cv::Point& p2)
{
    double a = static_cast<double>(p2.y - p1.y);
    double b = static_cast<double>(p1.x - p2.x);
    double norm = std::hypot(a, b);
    if (norm == 0.0) return {};
    double c = -a * p1.x - b * p1.y;
    return { a / norm, b / norm, c / norm };
}

inline double point_line_distance(const cv::Point& p, const LineABC& l)
{
    return std::abs(l.a * p.x + l.b * p.y + l.c);
}

inline bool line_intersection(const LineABC& l1, const LineABC& l2, cv::Point& out)
{
    double det = l1.a * l2.b - l2.a * l1.b;
    if (std::abs(det) < 1e-12) return false; // parallel
    double x = (l1.b * l2.c - l2.b * l1.c) / det;
    double y = (l2.a * l1.c - l1.a * l2.c) / det;
    out = cv::Point(cvRound(x), cvRound(y));
    return true;
}

inline int blur_kernel_radius(const cv::Mat& img)
{
    return (std::max(img.rows, img.cols) / 200);
}

// ------------------ funkce pro předzpracování snímků --------------------
static cv::Mat smooth_noise_gb(const cv::Mat& img)
{
    int s = blur_kernel_radius(img) * 2 + 1;
    double sigma = 0.3 * ((s - 1) / 2.0 - 1) + 0.8;
    cv::Mat out;
    cv::GaussianBlur(img, out, cv::Size(s, s), sigma, sigma, cv::BORDER_REPLICATE);
    return out;
}

static cv::Mat apply_sobel_filter(const cv::Mat& img)
{
    cv::Mat grad_x, grad_y, mag;
    cv::Sobel(img, grad_x, CV_64F, 1, 0, 3);
    cv::Sobel(img, grad_y, CV_64F, 0, 1, 3);
    cv::magnitude(grad_x, grad_y, mag);
    double maxv;
    cv::minMaxLoc(mag, nullptr, &maxv);
    cv::Mat mag8u;
    mag.convertTo(mag8u, CV_8U, 255.0 / (maxv + 1e-9));
    return mag8u;
}

static cv::Mat process_edge_image(cv::Mat img)
{
    // 1. nahradit 11 % nejjasnějších pixelů jejich minimem
    cv::Mat flat = img.reshape(1, 1); // jeden řádek
    std::vector<uchar> vals(flat.begin<uchar>(), flat.end<uchar>());
    std::nth_element(vals.begin(), vals.begin() + vals.size() * 89 / 100, vals.end());
    uchar thresh = vals[vals.size() * 89 / 100];
    uchar minBright = *std::min_element(vals.begin() + vals.size() * 89 / 100, vals.end());
    img.setTo(minBright, img >= thresh);

    // 2. normalizace histogramu zpět na [0,255]
    double minv, maxv;
    cv::minMaxLoc(img, &minv, &maxv);
    img.convertTo(img, CV_8U, 255.0 / std::max(1.0, maxv - minv), -minv * 255.0 / std::max(1.0, maxv - minv));

    // 3. práh >254
    cv::Mat bw = img > 254;
    bw.convertTo(bw, CV_8U, 255);
    return bw;
}

static cv::Mat apply_otsu_threshold(const cv::Mat& img)
{
    cv::Mat mask;
    cv::threshold(img, mask, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
    if (cv::mean(mask)[0] < 127) cv::bitwise_not(mask, mask);
    return mask;
}

static cv::Mat remove_small_components(const cv::Mat& mask, double minAreaPct = 0.15)
{
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask.clone(), contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    
    cv::Mat mask_filtered(mask.size(), CV_8U, cv::Scalar(255));
    
    double minArea = mask.total() * minAreaPct;
    for (const auto& c : contours)
        if (cv::contourArea(c) < minArea)
            cv::drawContours(mask_filtered, std::vector<std::vector<cv::Point>>{c}, -1, cv::Scalar(0), cv::FILLED);

    cv::Mat out;
    cv::bitwise_and(mask, mask_filtered, out);
    return out;
}

// -------------- pomocné funkce pro hledání hran -----------------------
static std::pair<double, double> estimate_slice_dims(const cv::Mat& edgeImg)
{
    if (edgeImg.empty() || cv::countNonZero(edgeImg) == 0)
        return { edgeImg.cols * 0.8, edgeImg.rows * 0.8 };

    // 1. Projekce energií přes řádky/sloupce
    cv::Mat projRow, projCol;
    cv::reduce(edgeImg, projRow, 1, cv::REDUCE_SUM, CV_32S);
    cv::reduce(edgeImg, projCol, 0, cv::REDUCE_SUM, CV_32S);

    // 2. Nalezení souřadnic extrémů
    int top    = 0; while (top  < projRow.rows && projRow.at<int>(top)  == 0) ++top;
    int bottom = projRow.rows - 1; while (bottom >= 0 && projRow.at<int>(bottom) == 0) --bottom;
    int left   = 0; while (left < projCol.cols && projCol.at<int>(left) == 0) ++left;
    int right  = projCol.cols - 1; while (right >= 0 && projCol.at<int>(right) == 0) --right;

    double approxW = right - left + 1;
    double approxH = bottom - top + 1;

    // 3. Zastropování na 80% rozměrů obrazu
    approxW = std::min<double>(edgeImg.cols * 0.8, approxW);
    approxH = std::min<double>(edgeImg.rows * 0.8, approxH);

    return { std::max(1.0, approxW), std::max(1.0, approxH) };
}

static cv::Mat cumulative_zero_counts(const cv::Mat& otsu)
{
    cv::Mat otsuRowcumsum(otsu.size(), CV_32S);
    #pragma omp parallel for
    for (int x = 0; x < otsu.cols; ++x) {
        int acc = 0;
        for (int y = 0; y < otsu.rows; ++y) {
            if (otsu.at<uchar>(y, x) == 0) ++acc;
            otsuRowcumsum.at<int>(y, x) = acc;
        }
    }
    return otsuRowcumsum;
}

static double hug_score_from_otsu(const LineABC& l, const cv::Mat& otsuRowcumsum, const cv::Size& sz, bool topEdge)
{
    const double A = l.a, B = l.b, C = l.c;
    // ošetření pro vertikální přímku
    if (std::abs(B) < 1e-9) return 1.0;

    const int h = sz.height, w = sz.width;
    double negSum = 0, totSum = 0;
    #pragma omp parallel for reduction(+:negSum,totSum)
    for (int x = 0; x < w; ++x) {
        int y = cvRound(-(A * x + C) / B);
        y = std::clamp(y, 0, h - 1);
        if (topEdge) {
            int neg = otsuRowcumsum.at<int>(y, x);
            int tot = y + 1;
            if (tot) { negSum += neg; totSum += tot; }
        } else {
            int negTotal = otsuRowcumsum.at<int>(h - 1, x);
            int negAbove = otsuRowcumsum.at<int>(y, x);
            int neg      = negTotal - negAbove;
            int tot      = (h - 1 - y);
            if (tot) { negSum += neg; totSum += tot; }
        }
    }
    return totSum == 0 ? 1.0 : negSum / totSum;
}

static bool find_edge_line_ransac(
    const cv::Mat& edge, const std::string& edgeType,
    const std::pair<double,double>& approxDims, const cv::Mat& otsuRowcumsum,
    LineABC& bestLine,
    int iterations, double slopeFactor, double hugFactor
)
{
    const int h = edge.rows, w = edge.cols;
    const auto [approxW, approxH] = approxDims;

    std::vector<cv::Point> points; points.reserve(cv::countNonZero(edge));
    for (int y = 0; y < h; ++y) {
        const uchar* row = edge.ptr<uchar>(y);
        for (int x = 0; x < w; ++x) if (row[x] > 0) points.emplace_back(x, y);
    }
    if (points.size() < 2) return false;

    // předvýpočet polovin semivzorku
    std::vector<cv::Point> ptsG1, ptsG2, relevant;
    double gaussMean = 0, gaussStd = 1;

    auto coordIndex = (edgeType == "left" || edgeType == "right") ? 0 : 1;

    if (edgeType == "left" || edgeType == "right") {
        double centerX = (edgeType == "left") ? (w - approxW) / 2.0 : w - (w - approxW) / 2.0;
        gaussMean = std::clamp(centerX, 0.0, static_cast<double>(w - 1));
        gaussStd  = std::max(1.0, approxW / 3.0);
        for (auto& p : points) if ((edgeType == "left" ? p.x < w * 0.5 : p.x >= w * 0.5)) relevant.push_back(p);
        for (auto& p : relevant) (p.y < h / 2) ? ptsG1.push_back(p) : ptsG2.push_back(p);
    } else {
        double centerY = (edgeType == "top") ? (h - approxH) / 2.0 : h - (h - approxH) / 2.0;
        gaussMean = std::clamp(centerY, 0.0, static_cast<double>(h - 1));
        gaussStd  = std::max(1.0, approxH / 3.0);
        for (auto& p : points) if ((edgeType == "top" ? p.y < h * 0.5 : p.y >= h * 0.5)) relevant.push_back(p);
        for (auto& p : relevant) (p.x < w / 2) ? ptsG1.push_back(p) : ptsG2.push_back(p);
    }
    //if (ptsG1.empty() || ptsG2.empty()) { ptsG1 = ptsG2 = relevant; }
    if (ptsG1.empty() || ptsG2.empty()) return false;

    // gaussovská rozdělení pro vzorkování bodů
    auto buildProbs = [&](const std::vector<cv::Point>& v) {
        std::vector<double> probs(v.size(), 1.0);
        double sum = 0;
        for (size_t i = 0; i < v.size(); ++i) {
            double x = (coordIndex == 0) ? v[i].x : v[i].y;
            probs[i] = std::exp(-0.5 * std::pow((x - gaussMean) / gaussStd, 2));
            sum += probs[i];
        }
        if (sum == 0) {
            std::fill(probs.begin(), probs.end(), 1.0 / v.size());
        } else {
            for (auto& p : probs) p /= sum;
        }
        return probs;
    };
    auto probsG1 = buildProbs(ptsG1);
    auto probsG2 = buildProbs(ptsG2);

    std::uniform_real_distribution<double> uni(0, 1);
    auto sampleWithProb = [&](const std::vector<cv::Point>& v, const std::vector<double>& probs) {
        double r = uni(rng), acc = 0.0;
        for (size_t i = 0; i < probs.size(); ++i) {
            acc += probs[i];
            if (r <= acc) return v[i];
        }
        return v.back();
    };

    const double inlierThresh = std::max(2.0, std::max(w, h) / 150.0);
    const double sigmaAng = M_PI * 20.0 / 180.0;  // ~20° tolerance

    double bestFitness = -1;

    for (int iter = 0; iter < iterations; ++iter) {
        cv::Point p1 = sampleWithProb(ptsG1, probsG1);
        cv::Point p2 = sampleWithProb(ptsG2, probsG2);
        LineABC line = line_params(p1, p2);
        if (line.a == 0 && line.b == 0) continue;

        std::vector<double> distances(relevant.size());
        #pragma omp parallel for
        for (size_t i = 0; i < relevant.size(); ++i) distances[i] = point_line_distance(relevant[i], line);
        int inliers = std::count_if(distances.begin(), distances.end(), [&](double d){ return d < inlierThresh; });

        double theta = std::abs(std::atan2(line.a, line.b));
        double dev = (edgeType == "left" || edgeType == "right") ? std::abs(theta - M_PI_2) : std::min(theta, M_PI - theta);
        double slopeScore = std::exp(-0.5 * std::pow(dev / sigmaAng, 2));
        double fitness = (1 - slopeFactor) * inliers + slopeFactor * (inliers * slopeScore);

        if (edgeType == "top" || edgeType == "bottom") {
            bool topEdge = (edgeType == "top");
            double hugScore = hug_score_from_otsu(line, otsuRowcumsum, edge.size(), topEdge);
            fitness = (1 - hugFactor) * fitness + hugFactor * (fitness * hugScore * hugScore);
        }

        if (fitness > bestFitness) {
            bestFitness = fitness;
            bestLine = line;
        }
    }
    return bestFitness > 0;
}

/* static bool find_slice_corners(
    const cv::Mat& edge, const cv::Mat& otsu, CornerSet& corners,
    int iterations, double slopeFactor, double hugFactor
)
{
    auto approxDims = estimate_slice_dims(edge);
    cv::Mat otsuRowcumsum = cumulative_zero_counts(otsu);

    LineABC leftE, rightE, topE, btmE;
    if (!find_edge_line_ransac(edge, "left",   approxDims, otsuRowcumsum, leftE,  iterations, slopeFactor, hugFactor)) return false;
    if (!find_edge_line_ransac(edge, "right",  approxDims, otsuRowcumsum, rightE, iterations, slopeFactor, hugFactor)) return false;
    if (!find_edge_line_ransac(edge, "top",    approxDims, otsuRowcumsum, topE,   iterations, slopeFactor, hugFactor)) return false;
    if (!find_edge_line_ransac(edge, "bottom", approxDims, otsuRowcumsum, btmE,   iterations, slopeFactor, hugFactor)) return false;

    if (!line_intersection(topE, leftE,  corners.tl)) return false;
    if (!line_intersection(topE, rightE, corners.tr)) return false;
    if (!line_intersection(btmE, rightE, corners.br)) return false;
    if (!line_intersection(btmE, leftE,  corners.bl)) return false;

    return true;
} */

// paralelizovaná verze
static bool find_slice_corners(
    const cv::Mat& edge, const cv::Mat& otsu, CornerSet& corners,
    int iterations, double slopeFactor, double hugFactor
)
{
    auto approxDims = estimate_slice_dims(edge);
    cv::Mat otsuRowcumsum = cumulative_zero_counts(otsu);

    LineABC leftE, rightE, topE, btmE;
    bool okLeft = false, okRight = false, okTop = false, okBtm = false;

    #pragma omp parallel sections default(none) shared( \
        edge, otsuRowcumsum, approxDims, iterations, slopeFactor, hugFactor, \
        okLeft, okRight, okTop, okBtm, leftE, rightE, topE, btmE \
    )
    {
        #pragma omp section
        okLeft = find_edge_line_ransac(
            edge, "left", approxDims, otsuRowcumsum,
            leftE, iterations, slopeFactor, hugFactor
        );

        #pragma omp section
        okRight = find_edge_line_ransac(
            edge, "right", approxDims, otsuRowcumsum,
            rightE, iterations, slopeFactor, hugFactor
        );

        #pragma omp section
        okTop = find_edge_line_ransac(
            edge, "top", approxDims, otsuRowcumsum,
            topE, iterations, slopeFactor, hugFactor
        );

        #pragma omp section
        okBtm = find_edge_line_ransac(
            edge, "bottom", approxDims, otsuRowcumsum,
            btmE, iterations, slopeFactor, hugFactor
        );
    }

    if (!okLeft || !okRight || !okTop || !okBtm)
        return false;

    if (!line_intersection(topE, leftE,  corners.tl)) return false;
    if (!line_intersection(topE, rightE, corners.tr)) return false;
    if (!line_intersection(btmE, rightE, corners.br)) return false;
    if (!line_intersection(btmE, leftE,  corners.bl)) return false;

    return true;
}

// ------------------------------- main ----------------------------------
int main(int argc, char** argv)
{
    // zpracování argumentů
    int iterations = 1000, seed = 42; double slopeFactor = 0.8, hugFactor = 0.8;
    std::vector<std::string> args(argv+1,argv+argc);
    size_t idx = 0;
    while (idx < args.size() && args[idx].rfind("--",0) == 0) {
        if (args[idx] == "--iterations" && idx+1 < args.size())
            iterations = std::stoi(args[++idx]);
        else if (args[idx] == "--slopeFactor" && idx+1 < args.size())
            slopeFactor = std::stod(args[++idx]);
        else if (args[idx] == "--hugFactor" && idx+1 < args.size())
            hugFactor = std::stod(args[++idx]);
        else if (args[idx] == "--seed" && idx+1 < args.size())
            seed = std::stoi(args[++idx]);
        else {
            std::cerr<<"Unknown option "<<args[idx]<<"\n";
            return 1;
        }
        idx++;
    }
    if (idx >= args.size()) {
        std::cerr<<"Usage: "<<argv[0]<<" [--iterations N] [--slopeFactor f] [--hugFactor f] <in> [out]\n";
        return 1;
    }
    std::string inPath = args[idx++], outPath = idx < args.size() ? args[idx] : "";
    rng.seed(seed);

    // načtení obrazu
    cv::Mat gray = cv::imread(inPath, cv::IMREAD_GRAYSCALE);
    if (gray.empty()) { std::cerr << "Cannot read image: " << inPath << "\n"; return 1; }

    // konverze na grayscale + normalizace
    double minv, maxv; cv::minMaxLoc(gray, &minv, &maxv);
    gray.convertTo(gray, CV_8U, 255.0 / (maxv + 1e-9));

    // předzpracování obrazu
    cv::Mat imgBlur  = smooth_noise_gb(gray);
    cv::Mat sobel    = apply_sobel_filter(imgBlur);
    cv::Mat edgeProc = process_edge_image(sobel);

    cv::Mat otsuBin  = apply_otsu_threshold(gray);
    cv::Mat otsuFilt = remove_small_components(otsuBin);
    int kernelH      = blur_kernel_radius(gray); kernelH += kernelH % 2 ? 1 : 0;
    cv::Mat kernel   = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(1, blur_kernel_radius(gray)+1));
    cv::Mat otsuDil  ; cv::morphologyEx(otsuFilt, otsuDil, cv::MORPH_DILATE, kernel);

    cv::Mat edgeMasked; cv::bitwise_and(edgeProc, edgeProc, edgeMasked, otsuDil);

    CornerSet cs;
    if (!find_slice_corners(edgeMasked, otsuDil, cs, iterations, slopeFactor, hugFactor)) {
        std::cerr << "Corner detection failed." << std::endl;
        return 2;
    }

    std::cout << cs.tl.x << " " << cs.tl.y << std::endl // TL
              << cs.tr.x << " " << cs.tr.y << std::endl // TR
              << cs.br.x << " " << cs.br.y << std::endl // BR
              << cs.bl.x << " " << cs.bl.y << std::endl;// BL

    if (!outPath.empty()) {
        cv::Mat display; cv::cvtColor(gray, display, cv::COLOR_GRAY2BGR);
        std::vector<cv::Point> poly{cs.tl, cs.tr, cs.br, cs.bl};

        // semitransparentní vykreslení polygonu
        cv::Mat overlay = display.clone();
        cv::polylines(overlay, poly, true, cv::Scalar(0,255,0), std::max(1, (int)(std::min(gray.rows, gray.cols)/200)));
        cv::addWeighted(display, 0.5, overlay, 0.5, 0, display);
        //cv::polylines(display, poly, true, cv::Scalar(0,255,0), std::max(1, (int)(std::min(gray.rows, gray.cols)/200)));
        //for (auto& p : poly) cv::circle(display, p, std::max(3, (int)(std::min(gray.rows, gray.cols)*0.01)), cv::Scalar(0,0,255), cv::FILLED);
        cv::imwrite(outPath, display);
    }

    return 0;
}
