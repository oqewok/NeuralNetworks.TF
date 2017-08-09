using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using OpenCvSharp.CPlusPlus;

namespace ImageConverter
{
	public static class ImageMaskCreator
	{
		public const int Width = 640;
		public const int Heigth = 480;

		public static string CreateStringMask(Rectangle[] boundBoxes)
		{
			var sb = new StringBuilder();
			var mask = new int[Width, Heigth];

			foreach(var box in boundBoxes)
			{
				for(int i = box.Left; i < box.Right; i++)
				{
					for(int j = box.Top; j < box.Bottom; j++)
					{
						mask[i, j] = 1; 
					}
				}
			}

			for(int i = 0; i < Width; i++)
			{
				for(int j = 0; j < Heigth; j++)
				{
					sb.Append(mask[i, j] + ",");
				}
			}

			return sb.ToString().Substring(0, sb.Length - 1);
		}
	}
}
