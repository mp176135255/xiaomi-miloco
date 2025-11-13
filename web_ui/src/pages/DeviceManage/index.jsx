/**
 * Copyright (C) 2025 Xiaomi Corporation
 * This software may be used and distributed according to the terms of the Xiaomi Miloco License Agreement.
 */

import React from 'react';
import { useTranslation } from 'react-i18next';
import { Header, Icon, PageContent } from '@/components';
import { DeviceList } from './components';
import { useDevices } from './hooks/useDevices';
import styles from './index.module.less';

/**
 * DeviceManage Page - Device management page for viewing and managing connected devices
 * 设备管理页面 - 用于查看和管理已连接设备的页面
 *
 * @returns {JSX.Element} Device management page component
 */
const DeviceManage = () => {
  const { t } = useTranslation();
  const { devices, loading, refreshDevices } = useDevices();

  return (
    <PageContent
      Header={(
        <Header
          title={t('home.menu.deviceManage')}
          rightContent={<div
            style={{
              display: 'flex',
              alignItems: 'center',
              cursor: 'pointer'
            }}
            onClick={refreshDevices}
          >
            <Icon
              name="refresh"
              size={15}
              style={{ color: 'var(--text-color)' }}
            />
            <span style={{ fontSize: '14px', color: 'var(--text-color)', marginLeft: '6px' }}>{t('common.refresh')}</span>
          </div>
          }
        />
      )}
      loading={loading}
      showEmptyContent={!loading && devices.length === 0}
      emptyContentProps={{
        description: t('deviceManage.noDevice'),
        imageStyle: { width: 72, height: 72 },
      }}
    >
      <DeviceList devices={devices} />
    </PageContent>
  );
};

export default DeviceManage;
