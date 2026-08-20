import { useState } from 'react';
import { motion } from 'framer-motion';
import { useAppConfig } from '@/contexts/AppConfigContext';

export const Greeting = () => {
  const { greeting, greetingImage, isLoading } = useAppConfig();
  const [imageFailed, setImageFailed] = useState(false);
  const showImage = Boolean(greetingImage) && !imageFailed;

  return (
    <div
      key="overview"
      className="mx-auto mb-6 flex size-full max-w-3xl flex-col justify-center px-4"
    >
      {isLoading ? null : (
        <div className="flex flex-col items-center gap-5">
          {showImage && greetingImage && (
            <motion.div
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: 10 }}
              className="flex justify-center"
            >
              <img
                src={greetingImage.url}
                alt={greetingImage.alt ?? ''}
                className="h-auto w-auto object-contain"
                style={{
                  maxWidth: greetingImage.maxWidth ?? '22rem',
                  maxHeight: greetingImage.maxHeight ?? '6rem',
                }}
                onError={() => setImageFailed(true)}
              />
            </motion.div>
          )}
          {greeting ? (
            <motion.div
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: 10 }}
              className="text-center font-semibold text-lg md:text-xl"
            >
              {greeting}
            </motion.div>
          ) : null}
        </div>
      )}
    </div>
  );
};
