import { IMAGE } from '$lib/constants';
import { MimeTypeImage } from '$lib/enums';

/**
 * Convert a HEIC/HEIF file to a compressed JPEG data URL
 * @param file - The HEIC/HEIF file to convert
 * @returns Promise resolving to JPEG data URL
 */
export async function heicFileToJpegDataURL(file: File | Blob): Promise<string> {
	const { heicTo } = await import('heic-to');
	const jpegBlob = await heicTo({
		blob: file,
		quality: IMAGE.HEIC_JPEG_QUALITY,
		type: MimeTypeImage.JPEG
	});

	return new Promise((resolve, reject) => {
		const reader = new FileReader();

		reader.onload = () => resolve(reader.result as string);
		reader.onerror = () => reject(reader.error);
		reader.readAsDataURL(jpegBlob);
	});
}

/**
 * Check if a MIME type represents a HEIC/HEIF image
 * @param mimeType - The MIME type to check
 * @returns True if the MIME type is image/heic or image/heif
 */
export function isHeicMimeType(mimeType: string): boolean {
	const normalized = mimeType.trim().toLowerCase();

	return normalized === MimeTypeImage.HEIC || normalized === MimeTypeImage.HEIF;
}
