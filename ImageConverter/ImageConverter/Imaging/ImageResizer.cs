using OpenCvSharp;
using OpenCvSharp.CPlusPlus;

namespace ImageConverter.Imaging
{
	public class ImageResizer
	{
		private const int ImgWidth = 50;
		private const double Ratio = 1.15;
		private const int ImgHeight = (int)(ImgWidth * (1 / Ratio));


		public void Resize(string filename)
		{
			var mat = new Mat(filename);

			if(mat.Width != ImgWidth || mat.Height != ImgHeight)
			{
				mat = GetResizedMat(mat, ImgWidth, ImgHeight);
				mat.SaveImage(filename);
			}
		}

		private Mat GetResizedMat(Mat src, int width, int height)
		{ 
			return src.Resize(
				new Size(width, height),
				fx: 1.0,
				fy: 1.0,
				interpolation: Interpolation.Cubic);
		}
	}
}
